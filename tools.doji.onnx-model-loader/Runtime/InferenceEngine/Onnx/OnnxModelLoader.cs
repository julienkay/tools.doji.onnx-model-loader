using System.IO;
using Unity.InferenceEngine;
using Doji.AI.InferenceEngine;
using Doji.ONNX;

namespace Doji.AI
{
    public static class OnnxModelLoader
    {
        public static Model Load(string filePath)
        {
            var converter = new ONNXModelConverter();
            return converter.Convert(filePath);
        }

        public static ONNXModelMetadata LoadMetadata(string filePath)
        {
            var converter = new ONNXModelConverter();
            return converter.LoadMetadata(filePath);
        }

        internal static ModelProto LoadModelMetadata(Stream stream)
        {
            var msg = new ModelProto();
            while (stream.Position < stream.Length)
            {
                uint tag = ProtoReader.ReadVarint32(stream);
                int field = (int)(tag >> 3);
                int wireType = (int)(tag & 7);
                switch (field)
                {
                    case 1:
                        msg.IrVersion = (long)ProtoReader.ReadVarint64(stream);
                        break;
                    case 2:
                        msg.ProducerName = ProtoReader.ReadString(stream);
                        break;
                    case 3:
                        msg.ProducerVersion = ProtoReader.ReadString(stream);
                        break;
                    case 4:
                        msg.Domain = ProtoReader.ReadString(stream);
                        break;
                    case 5:
                        msg.ModelVersion = (long)ProtoReader.ReadVarint64(stream);
                        break;
                    case 6:
                        msg.DocString = ProtoReader.ReadString(stream);
                        break;
                    case 14:
                        msg.MetadataProps.Add(StringStringEntryProto.Deserialize(new MemoryStream(ProtoReader.ReadBytes(stream))));
                        break;
                    case 7:
                    case 8:
                    case 20:
                    case 25:
                    case 26:
                        ProtoReader.SkipBytes(stream);
                        break;
                    default:
                        ProtoReader.SkipField(stream, wireType);
                        break;
                }
            }

            return msg;
        }
    }
}
