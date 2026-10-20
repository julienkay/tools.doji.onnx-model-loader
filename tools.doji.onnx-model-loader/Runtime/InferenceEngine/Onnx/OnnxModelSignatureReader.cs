using System;
using System.Collections.Generic;
using System.IO;
using Doji.ONNX;

namespace Doji.AI
{
    /// <summary>
    /// Reads lightweight ONNX graph signature data without loading model weights or node payloads.
    /// </summary>
    public static class OnnxModelSignatureReader
    {
        public static OnnxModelSignature ReadSignature(string filePath)
        {
            using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            return ReadSignature(stream);
        }

        public static OnnxModelSignature ReadSignature(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (!stream.CanSeek)
                throw new InvalidDataException("ONNX signature reading requires a seekable stream.");

            while (stream.Position < stream.Length)
            {
                uint tag = ProtoReader.ReadVarint32(stream);
                int field = (int)(tag >> 3);
                int wireType = (int)(tag & 7);
                if (field == 7)
                    return ReadGraphSignature(stream, wireType);

                SkipField(stream, wireType);
            }

            return new OnnxModelSignature();
        }

        static OnnxModelSignature ReadGraphSignature(Stream stream, int wireType)
        {
            RequireWireType(wireType, 2, "ModelProto.graph");

            long end = ReadLengthDelimitedEnd(stream);
            var inputs = new List<OnnxModelInputSignature>();
            var initializerNames = new HashSet<string>(StringComparer.Ordinal);

            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    switch (field)
                    {
                        case 5:
                            var initializerName = ReadInitializerName(stream, fieldWireType);
                            if (!string.IsNullOrEmpty(initializerName))
                                initializerNames.Add(initializerName);
                            break;
                        case 11:
                            var input = ReadValueInfo(stream, fieldWireType);
                            if (!string.IsNullOrEmpty(input.Name))
                                inputs.Add(input);
                            break;
                        default:
                            SkipField(stream, fieldWireType);
                            break;
                    }
                }
            }
            finally
            {
                stream.Position = end;
            }

            var filteredInputs = new List<OnnxModelInputSignature>();
            foreach (var input in inputs)
            {
                if (!initializerNames.Contains(input.Name))
                    filteredInputs.Add(input);
            }

            return new OnnxModelSignature { Inputs = filteredInputs.ToArray() };
        }

        static string ReadInitializerName(Stream stream, int wireType)
        {
            RequireWireType(wireType, 2, "GraphProto.initializer");

            long end = ReadLengthDelimitedEnd(stream);
            string name = null;
            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    if (field == 8)
                        name = ProtoReader.ReadString(stream);
                    else
                        SkipField(stream, fieldWireType);
                }
            }
            finally
            {
                stream.Position = end;
            }

            return name;
        }

        static OnnxModelInputSignature ReadValueInfo(Stream stream, int wireType)
        {
            RequireWireType(wireType, 2, "GraphProto.input");

            long end = ReadLengthDelimitedEnd(stream);
            var input = new OnnxModelInputSignature();
            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    switch (field)
                    {
                        case 1:
                            input.Name = ProtoReader.ReadString(stream);
                            break;
                        case 2:
                            ReadTypeProto(stream, fieldWireType, input);
                            break;
                        default:
                            SkipField(stream, fieldWireType);
                            break;
                    }
                }
            }
            finally
            {
                stream.Position = end;
            }

            return input;
        }

        static void ReadTypeProto(Stream stream, int wireType, OnnxModelInputSignature input)
        {
            RequireWireType(wireType, 2, "ValueInfoProto.type");

            long end = ReadLengthDelimitedEnd(stream);
            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    if (field == 1)
                        ReadTensorType(stream, fieldWireType, input);
                    else
                        SkipField(stream, fieldWireType);
                }
            }
            finally
            {
                stream.Position = end;
            }
        }

        static void ReadTensorType(Stream stream, int wireType, OnnxModelInputSignature input)
        {
            RequireWireType(wireType, 2, "TypeProto.tensor_type");

            long end = ReadLengthDelimitedEnd(stream);
            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    switch (field)
                    {
                        case 1:
                            input.ElementType = (int)ProtoReader.ReadVarint32(stream);
                            break;
                        case 2:
                            input.Dimensions = ReadTensorShape(stream, fieldWireType);
                            break;
                        default:
                            SkipField(stream, fieldWireType);
                            break;
                    }
                }
            }
            finally
            {
                stream.Position = end;
            }
        }

        static OnnxModelDimension[] ReadTensorShape(Stream stream, int wireType)
        {
            RequireWireType(wireType, 2, "TensorType.shape");

            long end = ReadLengthDelimitedEnd(stream);
            var dimensions = new List<OnnxModelDimension>();
            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    if (field == 1)
                        dimensions.Add(ReadDimension(stream, fieldWireType, dimensions.Count));
                    else
                        SkipField(stream, fieldWireType);
                }
            }
            finally
            {
                stream.Position = end;
            }

            return dimensions.ToArray();
        }

        static OnnxModelDimension ReadDimension(Stream stream, int wireType, int index)
        {
            RequireWireType(wireType, 2, "TensorShapeProto.dim");

            long end = ReadLengthDelimitedEnd(stream);
            var dimension = new OnnxModelDimension { Index = index };
            try
            {
                while (stream.Position < end)
                {
                    uint tag = ProtoReader.ReadVarint32(stream);
                    int field = (int)(tag >> 3);
                    int fieldWireType = (int)(tag & 7);
                    switch (field)
                    {
                        case 1:
                            dimension.DimValue = (long)ProtoReader.ReadVarint64(stream);
                            break;
                        case 2:
                            dimension.DimParam = ProtoReader.ReadString(stream);
                            break;
                        default:
                            SkipField(stream, fieldWireType);
                            break;
                    }
                }
            }
            finally
            {
                stream.Position = end;
            }

            return dimension;
        }

        static long ReadLengthDelimitedEnd(Stream stream)
        {
            ulong length = ProtoReader.ReadVarint64(stream);
            if (length > long.MaxValue)
                throw new InvalidDataException("Length-delimited protobuf field is too large.");

            long end = checked(stream.Position + (long)length);
            if (end > stream.Length)
                throw new EndOfStreamException();

            return end;
        }

        static void SkipField(Stream stream, int wireType)
        {
            switch (wireType)
            {
                case 0:
                    ProtoReader.ReadVarint64(stream);
                    break;
                case 1:
                    stream.Seek(8, SeekOrigin.Current);
                    break;
                case 2:
                    stream.Position = ReadLengthDelimitedEnd(stream);
                    break;
                case 5:
                    stream.Seek(4, SeekOrigin.Current);
                    break;
                default:
                    throw new InvalidDataException($"Unknown protobuf wire type: {wireType}.");
            }
        }

        static void RequireWireType(int actualWireType, int expectedWireType, string fieldName)
        {
            if (actualWireType != expectedWireType)
                throw new InvalidDataException($"{fieldName} has wire type {actualWireType}; expected {expectedWireType}.");
        }
    }

    [Serializable]
    public sealed class OnnxModelSignature
    {
        public OnnxModelInputSignature[] Inputs { get; set; } = Array.Empty<OnnxModelInputSignature>();
    }

    [Serializable]
    public sealed class OnnxModelInputSignature
    {
        public string Name { get; set; }
        public int ElementType { get; set; }
        public OnnxModelDimension[] Dimensions { get; set; } = Array.Empty<OnnxModelDimension>();
    }

    [Serializable]
    public sealed class OnnxModelDimension
    {
        public int Index { get; set; }
        public long? DimValue { get; set; }
        public string DimParam { get; set; }

        public bool IsDynamic => !DimValue.HasValue;
        public bool IsNamed => !string.IsNullOrWhiteSpace(DimParam);
    }
}
