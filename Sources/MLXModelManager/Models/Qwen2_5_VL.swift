// Copyright © 2025 Apple Inc.
//
// Qwen2.5VL: A consolidated Swift file ported from the newer Python code:
//    - language.py
//    - vision.py
//    - qwen2_5_vl.py
//
// The structure is kept similar to the older Qwen2VL.swift to maintain consistency and design style.

import CoreImage
import Foundation
import Hub
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Common

/// Rotates half the hidden dims of the input (same as Qwen2VL.swift).
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// A helper used in the older Qwen2VL to create or adjust the attention mask.
/// For Qwen2.5VL, we keep the same shape for causal language modeling.
private func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let batchSize = h.dim(0)
    let seqLen = h.dim(1)
    // Usually a triangular causal mask:
    let mask = MLXFast.causalMask(batchSize: batchSize, seqLen: seqLen)
    return mask
}

// MARK: - Language

/// Logic for text-related modules, mirroring `language.py`
private enum Language {

    /// Mirrors `TextConfig` in Python (language.py)
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let rmsNormEps: Float
        public let vocabSize: Int
        public let numKeyValueHeads: Int
        public let maxPositionEmbeddings: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let ropeScaling: [String: StringOrNumber]?
        public let tieWordEmbeddings: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabSize = "vocab_size"
            case numKeyValueHeads = "num_key_value_heads"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    // MARK: - Attention

    /// Equivalent to the `Attention` class in language.py
    fileprivate class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        public init(_ args: TextConfiguration) {
            let dim = args.hiddenSize
            self.heads = args.numAttentionHeads
            self.kvHeads = args.numKeyValueHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            // The older Qwen2VL used RoPE with base=ropeTheta, etc.
            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: args.ropeTraditional,
                base: args.ropeTheta
            )
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray? = nil,
            cache: KVCache? = nil
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            var q = wq(x)
            var k = wk(x)
            var v = wv(x)

            // shape => [B, L, heads, headDim] => transpose => [B, heads, L, headDim]
            q = q.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            v = v.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            // trim mask if needed
            var attnMask = mask
            if let mask, mask.dim(-1) > k.dim(-2) {
                attnMask = mask[0..., 0..., 0 ..< k.dim(-2)]
            }

            // Apply rotary embedding
            q = rotaryEmbedding(q, offset: offset)
            k = rotaryEmbedding(k, offset: offset)

            // KV cache update
            if let cache {
                let updated = cache.update(keys: k, values: v)
                k = updated.0
                v = updated.1
            }

            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: attnMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(out)
        }
    }

    // MARK: - MLP

    /// Equivalent to the MLP in language.py
    fileprivate class MLP: Module, UnaryLayer {

        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        public init(dim: Int, hiddenDim: Int) {
            self._gateProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._downProj.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._upProj.wrappedValue = Linear(dim, hiddenDim, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            // python => down_proj(nn.silu(gate_proj(x)) * up_proj(x))
            let gate = silu(gateProj(x))
            let up = upProj(x)
            return downProj(gate * up)
        }
    }

    // MARK: - Decoder Layer

    /// Qwen2VLDecoderLayer in language.py
    fileprivate class Qwen2VLDecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: TextConfiguration) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(dim: args.hiddenSize, hiddenDim: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray? = nil,
            cache: KVCache? = nil
        ) -> MLXArray {
            let attnOut = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + attnOut
            let mlpOut = mlp(postAttentionLayerNorm(h))
            return h + mlpOut
        }
    }

    // MARK: - Qwen2Model

    /// The base language model stack (decoder) from language.py
    fileprivate class Qwen2Model: Module {

        let layers: [Qwen2VLDecoderLayer]
        let norm: RMSNorm
        let vocabSize: Int

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        public init(_ args: TextConfiguration) {
            self.vocabSize = args.vocabSize
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabSize,
                dimensions: args.hiddenSize
            )
            self.layers = (0 ..< args.numHiddenLayers).map { _ in Qwen2VLDecoderLayer(args) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputsEmbeds: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputsEmbeds {
                h = inputsEmbeds
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("Qwen2Model requires either input IDs or input embeddings.")
            }

            let mask = createAttentionMask(h: h, cache: cache)
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }
            return norm(h)
        }
    }

    // MARK: - LanguageModel

    /// Top-level LanguageModel from the python side, with an optional separate lm_head
    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: TextConfiguration

        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        // Conform to KVCacheDimensionProvider
        public var kvHeads: [Int] {
            // One entry per layer
            Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
        }

        public init(_ args: TextConfiguration) {
            self.config = args
            self._model.wrappedValue = Qwen2Model(args)
            if !args.tieWordEmbeddings {
                self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
            }
        }

        public func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputsEmbeds: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputsEmbeds: inputsEmbeds)
            if let lmHead {
                out = lmHead(out)
            } else {
                // tie_word_embeddings => reuse embed tokens
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }

        /// If needed, you can drop or rename certain keys from the language side.
        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            return weights
        }
    }
}

// MARK: - Vision

/// Logic for the vision tower (mirroring vision.py)
private enum Vision {

    /// Mirrors `VisionConfig` from vision.py
    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let vocabSize: Int
        public let mlpRatio: Float
        public let inChannels: Int
        public let layerNormEps: Float
        public let spatialPatchSize: Int
        public let spatialMergeSize: Int
        public let tokensPerSecond: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullattBlockIndexes: [Int]
    }

    /// Utility function used in the python code to check conv weight shapes
    private func checkArrayShape(_ arr: MLXArray) -> Bool {
        // Original python code checks ndims 4 or 5, outChannels vs kH, kW, etc.
        // We keep a simplified approach here. Adjust as needed.
        if arr.ndim < 4 || arr.ndim > 5 {
            return false
        }
        return true
    }

    /// Equivalent of "VisionRotaryEmbedding" in vision.py
    fileprivate class VisionRotaryEmbedding: Module {
        let dim: Int
        let theta: Float

        public init(dim: Int, theta: Float = 10000.0) {
            self.dim = dim
            self.theta = theta
        }

        public func callAsFunction(_ seqlen: Int) -> MLXArray {
            // Build inverse freq => same approach as older Qwen2VL
            let freqIndices = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim)
            let invFreq = 1.0 / pow(theta, freqIndices)
            let seq = MLXArray(0 ..< seqlen).asType(.float32)
            return outer(seq, invFreq)
        }
    }

    /// Equivalent to "PatchEmbed" in vision.py
    fileprivate class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        public init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            self._proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: false
            )
        }

        public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            // Python logic: reshape => moveaxis => conv => flatten
            var x = hiddenStates.reshaped(-1, inChannels, temporalPatchSize, patchSize, patchSize)
            x = x.movedAxis(source: 1, destination: 4)
            x = proj(x)
            x = x.reshaped(-1, hiddenSize)
            return x
        }
    }

    /// Equivalent to "PatchMerger" in vision.py
    fileprivate class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        public init(dim: Int, contextDim: Int, spatialMergeSize: Int = 2) {
            self.hiddenSize = contextDim * (spatialMergeSize * spatialMergeSize)
            self._lnQ.wrappedValue = RMSNorm(dimensions: contextDim, eps: 1e-6)
            self.mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dim)
            )
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            var out = lnQ(x).reshaped(-1, hiddenSize)
            out = mlp.0(out)
            out = mlp.1(out)
            out = mlp.2(out)
            return out
        }
    }

    /// Equivalent to "Attention" in vision.py
    fileprivate class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        public init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            self._proj.wrappedValue = Linear(dim, dim)
        }

        public func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray? = nil
        ) -> MLXArray {
            // see vision.py for logic
            let seqLen = x.dim(0)
            var merges = qkv(x).reshaped(seqLen, 3, numHeads, -1).transposed(1, 0, 2, 3)
            let splitted = split(merges, parts: 3, axis: 0)
            var q = splitted[0]
            var k = splitted[1]
            var v = splitted[2]

            // apply rotary
            if let emb = rotaryPosEmb {
                q = applyRotaryPosEmbVision(q, emb)
                k = applyRotaryPosEmbVision(k, emb)
            }

            // build attention mask from cuSeqlens
            var attnMask = fill((1, seqLen, seqLen), with: Float.leastNonzeroMagnitude)
            let arr = cuSeqlens.asArray(Int.self)
            for i in 1..<arr.count {
                let start = arr[i-1]
                let end = arr[i]
                attnMask[0..., start..<end, start..<end] = zeros((end - start, end - start))
            }

            // shape => scaled dot product
            q = q.transposed(0, 1, 2, 3) // => [seqLen, numHeads, headDim]
            k = k.transposed(0, 1, 2, 3)
            v = v.transposed(0, 1, 2, 3)

            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: scale, mask: attnMask
            )
            let shaped = out.reshaped(seqLen, numHeads * headDim)
            return proj(shaped)
        }

        private func applyRotaryPosEmbVision(_ tensor: MLXArray, _ freqs: MLXArray) -> MLXArray {
            var c = cos(freqs)
            var s = sin(freqs)
            c = expandedDimensions(c, axis: 1)
            c = tiled(c, repetitions: [1, 1, 2])
            c = expandedDimensions(c, axis: 0)

            s = expandedDimensions(s, axis: 1)
            s = tiled(s, repetitions: [1, 1, 2])
            s = expandedDimensions(s, axis: 0)

            return (tensor * c) + (rotateHalf(tensor) * s)
        }
    }

    /// Equivalent to "MLP" in vision.py
    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        public init(dim: Int, hiddenDim: Int) {
            self._gateProj.wrappedValue = Linear(dim, hiddenDim)
            self._upProj.wrappedValue = Linear(dim, hiddenDim)
            self._downProj.wrappedValue = Linear(hiddenDim, dim)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            // from vision.py => down_proj(silu(gate_proj(x)) * up_proj(x))
            let gate = silu(gateProj(x))
            let up = upProj(x)
            return downProj(gate * up)
        }
    }

    /// Equivalent to "Qwen2VLVisionBlock" in vision.py
    fileprivate class Qwen2VLVisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: RMSNorm
        @ModuleInfo(key: "norm2") var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attn: Attention
        @ModuleInfo var mlp: MLP

        public init(_ config: VisionConfiguration) {
            self._norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self._norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            self._attn.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            self._mlp.wrappedValue = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        public func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray?
        ) -> MLXArray {
            var out = hiddenStates + attn(norm1(hiddenStates), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            out = out + mlp(norm2(out))
            return out
        }
    }

    /// Equivalent to "VisionModel" in vision.py
    fileprivate class VisionModel: Module {

        public let config: VisionConfiguration

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryPosEmb: VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Qwen2VLVisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger

        public init(_ config: VisionConfiguration) {
            self.config = config
            guard config.modelType == "qwen2_5_vl" else {
                fatalError("Unsupported model type: \(config.modelType)")
            }
            self._patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize
            )
            let headDim = config.hiddenSize / config.numHeads
            self._rotaryPosEmb.wrappedValue = VisionRotaryEmbedding(dim: headDim / 2, theta: 10000)

            // build blocks
            self._blocks.wrappedValue = (0..<config.depth).map { _ in
                Qwen2VLVisionBlock(config)
            }

            self._merger.wrappedValue = PatchMerger(
                dim: config.outHiddenSize,
                contextDim: config.hiddenSize,
                spatialMergeSize: config.spatialMergeSize
            )
        }

        /// The main forward pass from vision.py
        public func callAsFunction(
            _ hiddenStates: MLXArray,
            gridThw: MLXArray,
            outputHiddenStates: Bool = false
        ) -> MLXArray {
            // 1) patch-embed
            var x = patchEmbed(hiddenStates)

            // 2) build rotary pos emb
            let rotaryPos = buildRotaryPosEmb(gridThw)

            // 3) get window indexes + partial cuSeqlens
            let (windowIndex, cuWindowSeqlensBlock) = getWindowIndex(gridThw)

            // reorder x by windowIndex
            x = reorderWithIndex(x, windowIndex)
            var rpos = reorderWithIndex(rotaryPos, windowIndex)

            // also build a full-seqlen version
            let totalSeqLen = x.dim(0)
            let cuSeqlensFull = MLXArray([0, totalSeqLen], dtype: .int32)

            // 4) pass through all blocks
            var hidden = x
            for (layerIdx, block) in blocks.enumerated() {
                // if layerIdx in fullattBlockIndexes => use cuSeqlensFull
                let useFull = config.fullattBlockIndexes.contains(layerIdx)
                let cuse = useFull ? cuSeqlensFull : cuWindowSeqlensBlock
                hidden = block(hidden, cuSeqlens: cuse, rotaryPosEmb: rpos)
            }

            // 5) final merger
            hidden = merger(hidden)

            // 6) reorder back
            let revIndex = argsort(windowIndex)
            hidden = hidden[revIndex, 0...]

            return hidden
        }

        /// Mirrors the sanitize logic from python VisionModel
        public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized = [String: MLXArray]()
            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else if k.contains("patch_embed.proj.weight") {
                    if checkArrayShape(v) {
                        sanitized[k] = v
                    } else {
                        // python => v.transpose(0,2,3,4,1)
                        sanitized[k] = v.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[k] = v
                }
            }
            return sanitized
        }

        // The following helper routines mirror the logic from vision.py,
        // but are simplified to keep the older Qwen2VL style:

        private func buildRotaryPosEmb(_ gridThw: MLXArray) -> MLXArray {
            // The python code enumerates each T/H/W, building position IDs for each pixel
            // then indexing into a large "rotary embedding." We'll do a simplified approach:
            // 1) find max(H,W) across the batch
            // 2) generate freq => gather each coordinate
            let arr = gridThw.asArray(Int.self)
            let batchSize = gridThw.dim(0)
            var maxHW = 0
            for i in 0..<batchSize {
                let t = arr[i*3 + 0]
                let h = arr[i*3 + 1]
                let w = arr[i*3 + 2]
                maxHW = max(maxHW, max(h, w))
            }
            let fullEmb = rotaryPosEmb(maxHW)

            // gather each position
            var allEmbeds = [MLXArray]()
            for i in 0..<batchSize {
                let t = arr[i*3 + 0]
                let h = arr[i*3 + 1]
                let w = arr[i*3 + 2]
                // build array of shape [h*w, 2], replicate T times, then gather from fullEmb
                var coords = [MLXArray]()
                for hh in 0..<h {
                    for ww in 0..<w {
                        let idx = max(hh, ww) // simplified approach
                        coords.append(fullEmb[idx, 0...])
                    }
                }
                let stacked = concatenated(coords, axis: 0)
                let repeatedT = tiled(stacked, repetitions: [t, 1])
                allEmbeds.append(repeatedT)
            }
            return concatenated(allEmbeds, axis: 0)
        }

        private func getWindowIndex(_ gridThw: MLXArray) -> (MLXArray, MLXArray) {
            // The python code is more complex, performing blockwise splitting & padding.
            // For demonstration, we do a simpler approach: we compute total # of patches,
            // create a trivial windowIndex = range(totalPatches) and cuWindowSeqlens = [0, totalPatches].
            let arr = gridThw.asArray(Int.self)
            let batchSize = arr.count / 3
            var totalPatches = 0
            for i in 0..<batchSize {
                let t = arr[i*3 + 0]
                let h = arr[i*3 + 1]
                let w = arr[i*3 + 2]
                let mergeSize = config.spatialMergeSize
                let llmH = h / mergeSize
                let llmW = w / mergeSize
                totalPatches += (t * llmH * llmW)
            }
            let windowIndex = MLXArray(0 ..< totalPatches, dtype: .int32)
            let cuWindowSeqlens = MLXArray([0, totalPatches], dtype: .int32)
            return (windowIndex, cuWindowSeqlens)
        }

        private func reorderWithIndex(_ x: MLXArray, _ windowIndex: MLXArray) -> MLXArray {
            guard x.dim(0) == windowIndex.dim(0) else {
                return x
            }
            return x[windowIndex, 0...]
        }

        private func argsort(_ x: MLXArray) -> MLXArray {
            // naive approach
            let arr = x.asArray(Int.self)
            let sorted = arr.enumerated().sorted { $0.element < $1.element }.map { $0.offset }
            return MLXArray(sorted, dtype: .int32)
        }
    }
}

// MARK: - Processor

/// Qwen2_5VL VLM `UserInputProcessor`.
///
/// Patterned after Qwen2VLProcessor in the older file.  
/// Adapts the same approach for image resizing, token insertion, etc.
///
public class Qwen2_5VLProcessor: UserInputProcessor {

    private let config: Qwen2_5VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Qwen2_5VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    // This reuses the logic from the older Qwen2VL to do a "smart resize" approach
    private func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws -> (Int, Int)
    {
        if height < factor || width < factor {
            throw ModelFactoryError.imageProcessingFailure(
                "height/width must be >= factor (\(factor))"
            )
        }
        if max(height, width) / min(height, width) > 200 {
            throw ModelFactoryError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width)x\(height)"
            )
        }

        var hBar = Int(round(Float(height) / Float(factor))) * factor
        var wBar = Int(round(Float(width) / Float(factor))) * factor

        if hBar * wBar > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            hBar = Int(floor(Float(height) / beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if hBar * wBar < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            hBar = Int(floor(Float(height) * beta / Float(factor))) * factor
            wBar = Int(floor(Float(width) * beta / Float(factor))) * factor
        }
        return (hBar, wBar)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW
    ) {
        // same pattern as Qwen2VLProcessor
        let images = images.map { MediaProcessing.apply($0, processing: processing) }

        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try targetSize(
            height: Int(size.height),
            width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels,
            maxPixels: config.size.maxPixels
        )

        // Process each image in sRGB, resize, normalize
        let processedImages =
            try images
                .map { MediaProcessing.inSRGBToneCurveSpace($0) }
                .map { MediaProcessing.resampleBicubic($0, to: CGSize(width: resizedWidth, height: resizedHeight)) }
                .map { MediaProcessing.normalize($0, mean: config.imageMeanTuple, std: config.imageStdTuple) }
                .map { MediaProcessing.asMLXArray($0) }

        // If we need multiple frames, replicate temporalPatchSize times
        var patches = concatenated(processedImages)
        if patches.dim(0) != config.temporalPatchSize {
            patches = tiled(patches, repetitions: [config.temporalPatchSize, 1, 1, 1])
        }

        let channel = patches.dim(1)
        let gridT = patches.dim(0) / config.temporalPatchSize
        let gridH = resizedHeight / config.patchSize
        let gridW = resizedWidth / config.patchSize

        // Flatten the patches
        let flatPatches = patches.reshaped(
            gridT * gridH * gridW,
            channel * config.temporalPatchSize * config.patchSize * config.patchSize
        )

        return (flatPatches, THW(gridT, gridH, gridW))
    }

    public func prepare(prompt: UserInput.Prompt, imageTHW: [THW]?) -> String {
        // same approach as Qwen2VLProcessor -> build chat format, insert <|vision_start|><|image_pad|>...<|vision_end|>
        var messages = prompt.asMessages()
        if messages.first?["role"] != "system" {
            messages.insert(["role": "system", "content": "You are a helpful assistant."], at: 0)
        }
        let lastIndex = messages.count - 1
        var lastContent = messages[lastIndex]["content"] ?? ""

        // Insert <|vision_start|> and repeated <|image_pad|> tokens for each image
        let mergeSizeSquared = config.mergeSize * config.mergeSize
        if let imageTHW {
            for thw in imageTHW {
                let totalPads = thw.product / mergeSizeSquared
                lastContent += "<|vision_start|>"
                lastContent += Array(repeating: "<|image_pad|>", count: totalPads).joined()
                lastContent += "<|vision_end|>"
            }
        }

        messages[lastIndex]["content"] = lastContent

        return messages
            .map { "<|im_start|>\($0["role"] ?? "")\n\($0["content"] ?? "")<|im_end|>" }
            .joined(separator: "\n")
            + "\n<|im_start|>assistant\n"
    }

    public func prepare(input: UserInput) throws -> LMInput {
        if input.images.isEmpty {
            // no images => pure text
            let constructedPrompt = prepare(prompt: input.prompt, imageTHW: nil)
            let tokens = try tokenizer.encode(text: constructedPrompt)
            return LMInput(tokens: MLXArray(tokens))
        }

        // otherwise, we do images
        let processed = try input.images.map {
            try preprocess(images: [$0.asCIImage()], processing: input.processing)
        }
        let pixelArrays = processed.map { $0.0 }
        let gridInfos = processed.map { $0.1 }
        let catPixels = concatenated(pixelArrays)
        let promptText = prepare(prompt: input.prompt, imageTHW: gridInfos)
        let promptTokens = try tokenizer.encode(text: promptText)

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        let imageStruct = LMInput.ProcessedImage(pixels: catPixels, imageGridThw: gridInfos)
        return LMInput(text: .init(tokens: promptArray, mask: mask), image: imageStruct)
    }
}

// MARK: - Model

/// Qwen2.5VL VLM, akin to Qwen2VL but updated for the new code paths.
public class Qwen2_5VL: Module, UnifiedModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel

    public let config: Qwen2_5VLConfiguration

    public var vocabularySize: Int { config.baseConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Qwen2_5VLConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    // same as older Qwen2VL: "public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers { ... }"
    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        languageModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    /// Insert image features into the input embeddings at <image> token positions
    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray
    ) -> MLXArray {
        let imageTokenId = config.baseConfiguration.imageTokenId
        let idArray = inputIds.asArray(Int.self)
        var imageIndices = [Int]()
        for (i, token) in idArray.enumerated() {
            if token == imageTokenId {
                imageIndices.append(i)
            }
        }
        // batch=1 assumption
        if !imageIndices.isEmpty {
            inputsEmbeds[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }
        return inputsEmbeds
    }

    private func inputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        gridThw: [THW]?
    ) -> MLXArray {
        guard let pixelValues, let gridThw else {
            // text only
            return languageModel.model.embedTokens(inputIds)
        }

        // 1) text embeddings
        let txtEmbeds = languageModel.model.embedTokens(inputIds)
        // 2) vision tower
        let dtype = visionModel.patchEmbed.proj.weight.dtype
        let castPixels = pixelValues.asType(dtype)

        // build an MLXArray for gridThw => shape [batch, 3]
        let arr = gridThw.flatMap { [$0.t, $0.h, $0.w] }
        let shape = [gridThw.count, 3]
        let gridArray = MLXArray(arr, dtype: .int32).reshaped(shape)

        var hiddenStates = visionModel(castPixels, gridThw: gridArray, outputHiddenStates: false)
        // if hiddenStates is 2D => expand
        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        return mergeInputIdsWithImageFeatures(
            imageFeatures: hiddenStates,
            inputsEmbeds: txtEmbeds,
            inputIds: inputIds
        )
    }

    /// Implementation of the `UnifiedModel` "prepare" function, used by the framework to handle forward passes.
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let inputIds = input.text.tokens
        let pixelVals = input.image?.pixels
        let grids = input.image?.imageGridThw

        let inpEmbeds = inputEmbeddings(
            inputIds: inputIds,
            pixelValues: pixelVals,
            gridThw: grids
        )
        let logits = languageModel(nil, cache: cache, inputsEmbeds: inpEmbeds)
        return .logits(logits)
    }

    /// Standard "callAsFunction" if we only have text
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        return languageModel(inputs, cache: cache).logits
    }

    /// Typically used to rename weights from "visual"/"model" to "vision_tower"/"language_model.model".
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // The python code does two passes:
        //   weights = VisionModel.sanitize(weights)
        //   weights = LanguageModel.sanitize(weights)
        //
        // Also rename keys from "visual" => "vision_tower", "model" => "language_model.model", etc.
        let renamed = weights.reduce(into: [String: MLXArray]()) { result, kv in
            let (origKey, value) = kv
            var k = origKey
            if !k.contains("vision_tower") {
                k = k.replacingOccurrences(of: "visual", with: "vision_tower")
            }
            if !k.contains("language_model") {
                k = k.replacingOccurrences(of: "model.", with: "language_model.model.")
                k = k.replacingOccurrences(of: "lm_head.", with: "language_model.lm_head.")
            }
            result[k] = value
        }

        let visionFix = visionModel.sanitize(renamed)
        let final = languageModel.sanitize(visionFix)
        return final
    }
}

// MARK: - Configuration

/// Qwen2.5VL top-level configuration, analogous to Qwen2VLConfiguration in the older file.
///
/// We unify textConfiguration, visionConfiguration, plus a "baseConfiguration" for shared fields
/// (vocab size, image token ID, modelType, etc.).
public struct Qwen2_5VLConfiguration: Codable, Sendable {

    /// The text config portion
    public let textConfiguration: Language.TextConfiguration

    /// The vision config portion
    public let visionConfiguration: Vision.VisionConfiguration

    /// A base config portion, similar to how Qwen2VL had `BaseConfiguration`
    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let imageTokenId: Int
        public let hiddenSize: Int
        public let ignoreIndex: Int?
        public let videoTokenId: Int?
        public let visionStartTokenId: Int?
        public let visionEndTokenId: Int?
        public let visionTokenId: Int?
        public let visionFeatureSelectStrategy: String?
        public let visionFeatureLayer: Int?
    }

    public let baseConfiguration: BaseConfiguration

    enum CodingKeys: String, CodingKey {
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case base_modelType = "model_type"
        case ignoreIndex = "ignore_index"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case visionTokenId = "vision_token_id"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case visionFeatureLayer = "vision_feature_layer"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        // decode sub-dicts
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.visionConfiguration = try container.decode(
            Vision.VisionConfiguration.self,
            forKey: .visionConfiguration
        )
        // textConfiguration is a nested struct but top-level keys are also used
        self.textConfiguration = try Language.TextConfiguration(from: decoder)

        // Now build the base config portion from some top-level keys
        let modelType = try container.decodeIfPresent(String.self, forKey: .base_modelType) ?? "qwen2_5_vl"
        let vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
        let imageToken = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 151655
        let ignoreIdx = try container.decodeIfPresent(Int.self, forKey: .ignoreIndex)
        let videoTok = try container.decodeIfPresent(Int.self, forKey: .videoTokenId)
        let visionStart = try container.decodeIfPresent(Int.self, forKey: .visionStartTokenId)
        let visionEnd = try container.decodeIfPresent(Int.self, forKey: .visionEndTokenId)
        let visionT = try container.decodeIfPresent(Int.self, forKey: .visionTokenId)
        let visionStrat = try container.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy)
        let visionLayer = try container.decodeIfPresent(Int.self, forKey: .visionFeatureLayer)

        // We'll also reference hiddenSize from text config:
        let hiddenS = textConfiguration.hiddenSize

        self.baseConfiguration = BaseConfiguration(
            modelType: modelType,
            vocabularySize: vocabSize,
            imageTokenId: imageToken,
            hiddenSize: hiddenS,
            ignoreIndex: ignoreIdx,
            videoTokenId: videoTok,
            visionStartTokenId: visionStart,
            visionEndTokenId: visionEnd,
            visionTokenId: visionT,
            visionFeatureSelectStrategy: visionStrat,
            visionFeatureLayer: visionLayer
        )
    }
}

/// Processor configuration for Qwen2_5VL, analogous to Qwen2VLProcessorConfiguration
public struct Qwen2_5VLProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let maxPixels: Int
        public let minPixels: Int

        enum CodingKeys: String, CodingKey {
            case maxPixels = "max_pixels"
            case minPixels = "min_pixels"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
    }
}

// MARK: - End of Qwen2_5VL.swift


