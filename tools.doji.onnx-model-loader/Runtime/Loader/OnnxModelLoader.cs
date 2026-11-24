using System.IO;
using Unity.InferenceEngine;
using Doji.AI.InferenceEngine;
using Doji.ONNX;

namespace Doji.AI
{
    public static class OnnxModelLoader
    {
        public static Model Load(string filePath) => Load(filePath, null);

        /// <summary>
        /// Loads an ONNX model and quantizes its weights while loading, one weight at a time, so a model larger than
        /// the memory can load. The weights are those <see cref="ModelQuantizer.QuantizeWeights"/> quantizes, except
        /// rank-1 tensors such as biases. Null keeps float32 weights.
        /// </summary>
        public static Model Load(string filePath, QuantizationType? quantization)
        {
            var converter = new ONNXModelConverter { Quantization = quantization };
            return converter.Convert(filePath);
        }

        /// <summary>
        /// Converts an ONNX model to a .sentis file. Unlike <see cref="Load(string, QuantizationType?)"/> and
        /// <see cref="ModelWriter.Save(string, Model)"/>, it never holds all float32 weights in memory: they go from
        /// the ONNX file to the .sentis file one chunk at a time.
        /// </summary>
        public static void Save(string filePath, string sentisPath, QuantizationType? quantization = null) =>
            Save(filePath, sentisPath, quantization, null);

        /// <summary>
        /// Like <see cref="Save(string, string, QuantizationType?)"/>, and calls <paramref name="beforeSave"/> with the
        /// converted model before it is written, for example to fix input shapes. The constants of the float32 weights
        /// have no data yet: change only the layers, inputs and outputs.
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
