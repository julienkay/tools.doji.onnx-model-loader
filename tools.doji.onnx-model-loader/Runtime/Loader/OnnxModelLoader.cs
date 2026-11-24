using System.IO;
using Unity.InferenceEngine;
using Doji.AI.InferenceEngine;
using Doji.ONNX;

namespace Doji.AI
{
    public static class OnnxModelLoader
    {
        /// <summary>
        /// Loads an ONNX file into a model with float32 weights. To load a .sentis file, use
        /// <see cref="ModelLoader.Load(string)"/> of Inference Engine.
        /// </summary>
        public static Model Load(string filePath) => Load(filePath, null);

        /// <summary>
        /// Loads an ONNX file and quantizes its weights while loading. Each float32 weight is in memory only while it
        /// is quantized, so a model whose float32 weights do not fit in memory can load, if its quantized weights fit.
        /// The quantized weights are those <see cref="ModelQuantizer.QuantizeWeights"/> quantizes, except rank-1
        /// tensors such as biases. Null keeps float32 weights.
        /// </summary>
        public static Model Load(string filePath, QuantizationType? quantization)
        {
            var converter = new ONNXModelConverter { Quantization = quantization };
            return converter.Convert(filePath);
        }

        /// <summary>
        /// Converts an ONNX file to a .sentis file, which <see cref="ModelLoader.Load(string)"/> loads later with no
        /// conversion. With <paramref name="quantization"/>, the file holds the quantized weights. Unlike
        /// <see cref="Load(string, QuantizationType?)"/> and <see cref="ModelWriter.Save(string, Model)"/>, it never
        /// holds all weights in memory, quantized or not: each weight is read from the ONNX file, quantized, and
        /// written to the .sentis file, one weight at a time.
        /// </summary>
        public static void Save(string filePath, string sentisPath, QuantizationType? quantization = null) =>
            Save(filePath, sentisPath, quantization, null);

        /// <summary>
        /// Like <see cref="Save(string, string, QuantizationType?)"/>, and calls <paramref name="beforeSave"/> with the
        /// converted model before it is written, for example to fix input shapes. The constants of the weights have no
        /// data yet: change only the layers, inputs and outputs.
        /// </summary>
        public static void Save(string filePath, string sentisPath, QuantizationType? quantization, System.Action<Model> beforeSave)
        {
            var converter = new ONNXModelConverter { Quantization = quantization };
            converter.Save(filePath, sentisPath, beforeSave);
        }

        public static ONNXModelMetadata LoadMetadata(string filePath)
        {
            var converter = new ONNXModelConverter();
            return converter.LoadMetadata(filePath);
        }

        /// <summary>
        /// Reads the top-level <c>ModelProto</c> fields and seeks past the graph, opset imports,
        /// training info, functions, and device configurations, so no payload is loaded.
        /// </summary>
        internal static ModelProto LoadModelMetadata(Stream stream)
        {
            stream = ProtoReader.EnsureSeekable(stream);
            long end = stream.Length;
            var msg = new ModelProto();
            while (stream.Position < end)
            {
                uint tag = ProtoReader.ReadTag(stream);
                int field = (int)(tag >> 3);
                int wireType = (int)(tag & 7);
                switch (field)
                {
                    case 1 when wireType == 0:
                        msg.IrVersion = ProtoReader.ReadInt64(stream);
                        break;
                    case 2 when wireType == 2:
                        msg.ProducerName = ProtoReader.ReadString(stream, end);
                        break;
                    case 3 when wireType == 2:
                        msg.ProducerVersion = ProtoReader.ReadString(stream, end);
                        break;
                    case 4 when wireType == 2:
                        msg.Domain = ProtoReader.ReadString(stream, end);
                        break;
                    case 5 when wireType == 0:
                        msg.ModelVersion = ProtoReader.ReadInt64(stream);
                        break;
                    case 6 when wireType == 2:
                        msg.DocString = ProtoReader.ReadString(stream, end);
                        break;
                    case 14 when wireType == 2:
                    {
                        long entryEnd = ProtoReader.ReadLengthDelimitedEnd(stream, end);
                        var entry = new StringStringEntryProto();
                        entry.MergeFrom(stream, entryEnd);
                        msg.MetadataProps.Add(entry);
                        break;
                    }
                    default:
                        // Includes graph (7), opset_import (8), training_info (20), functions (25)
                        // and configuration (26): skipped by seeking, never read.
                        ProtoReader.SkipField(stream, wireType, end);
                        break;
                }
            }

            return msg;
        }
    }
}
