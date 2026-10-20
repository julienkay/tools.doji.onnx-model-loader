using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Doji.AI;
using NUnit.Framework;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    public class OnnxModelSignatureReaderTests
    {
        [Test]
        public void ReadSignature_ReturnsInputsAndDimensions()
        {
            var model = BuildModel(graph =>
            {
                WriteGraphInput(graph, "input", 1,
                    DimValue(1),
                    DimParam("batch"),
                    UnknownDim(),
                    DimValue(5000000000));
            });

            var signature = ReadSignature(model);

            Assert.That(signature.Inputs, Has.Length.EqualTo(1));
            var input = signature.Inputs[0];
            Assert.That(input.Name, Is.EqualTo("input"));
            Assert.That(input.ElementType, Is.EqualTo(1));
            Assert.That(input.Dimensions, Has.Length.EqualTo(4));
            Assert.That(input.Dimensions[0].DimValue, Is.EqualTo(1));
            Assert.That(input.Dimensions[1].DimParam, Is.EqualTo("batch"));
            Assert.That(input.Dimensions[2].IsDynamic, Is.True);
            Assert.That(input.Dimensions[3].DimValue, Is.EqualTo(5000000000));
        }

        [Test]
        public void ReadSignature_ExcludesInitializerInputs()
        {
            var model = BuildModel(graph =>
            {
                WriteInitializer(graph, "weight");
                WriteGraphInput(graph, "data", 1, DimValue(1));
                WriteGraphInput(graph, "weight", 1, DimValue(16));
            });

            var signature = ReadSignature(model);

            Assert.That(signature.Inputs, Has.Length.EqualTo(1));
            Assert.That(signature.Inputs[0].Name, Is.EqualTo("data"));
        }

        [Test]
        public void ReadSignature_SkipsLargeInitializerRawData()
        {
            var model = BuildModel(graph =>
            {
                WriteInitializer(graph, "weight", rawDataBytes: 1024 * 1024);
                WriteGraphInput(graph, "input", 1, DimParam("batch"));
            });

            using var stream = new ThrowOnReadRangeStream(model.Bytes, model.ForbiddenReadRanges);
            var signature = OnnxModelSignatureReader.ReadSignature(stream);

            Assert.That(signature.Inputs, Has.Length.EqualTo(1));
            Assert.That(signature.Inputs[0].Name, Is.EqualTo("input"));
        }

        [Test]
        public void ReadSignature_SkipsLargeNodePayload()
        {
            var model = BuildModel(graph =>
            {
                WriteLargeNodePayload(graph, 1024 * 1024);
                WriteGraphInput(graph, "input", 1, DimValue(1));
            });

            using var stream = new ThrowOnReadRangeStream(model.Bytes, model.ForbiddenReadRanges);
            var signature = OnnxModelSignatureReader.ReadSignature(stream);

            Assert.That(signature.Inputs, Has.Length.EqualTo(1));
            Assert.That(signature.Inputs[0].Name, Is.EqualTo("input"));
        }

        [Test]
        public void ReadSignature_DoesNotRequireUnityEditorOrAssetImport()
        {
            var model = BuildModel(graph =>
            {
                WriteGraphInput(graph, "input", 1, DimValue(1));
            });

            using var stream = new MemoryStream(model.Bytes);
            Assert.DoesNotThrow(() => OnnxModelSignatureReader.ReadSignature(stream));
        }

        [Test]
        public void ReadSignature_WithNoGraphInputs_ReturnsEmptySignature()
        {
            var model = BuildModel(_ => { });

            var signature = ReadSignature(model);

            Assert.That(signature.Inputs, Is.Empty);
        }

        [Test]
        public void ReadSignature_PreservesSymbolicDimParam()
        {
            var model = BuildModel(graph =>
            {
                WriteGraphInput(graph, "input", 1, DimParam("batch_size"));
            });

            var dimension = ReadSignature(model).Inputs[0].Dimensions[0];

            Assert.That(dimension.DimValue.HasValue, Is.False);
            Assert.That(dimension.DimParam, Is.EqualTo("batch_size"));
            Assert.That(dimension.IsDynamic, Is.True);
            Assert.That(dimension.IsNamed, Is.True);
        }

        [Test]
        public void ReadSignature_UnknownDimensionIsDynamic()
        {
            var model = BuildModel(graph =>
            {
                WriteGraphInput(graph, "input", 1, UnknownDim());
            });

            var dimension = ReadSignature(model).Inputs[0].Dimensions[0];

            Assert.That(dimension.DimValue.HasValue, Is.False);
            Assert.That(dimension.DimParam, Is.Null);
            Assert.That(dimension.IsDynamic, Is.True);
            Assert.That(dimension.IsNamed, Is.False);
        }

        static OnnxModelSignature ReadSignature(BuiltModel model)
        {
            using var stream = new MemoryStream(model.Bytes);
            return OnnxModelSignatureReader.ReadSignature(stream);
        }

        static BuiltModel BuildModel(Action<GraphWriter> writeGraph)
        {
            var graph = new GraphWriter();
            writeGraph(graph);

            using var model = new MemoryStream();
            var forbiddenReadRanges = new List<ByteRange>();
            WriteMessageField(model, 7, graph.Bytes, graph.ForbiddenReadRanges, forbiddenReadRanges);
            return new BuiltModel(model.ToArray(), forbiddenReadRanges);
        }

        static void WriteGraphInput(GraphWriter graph, string name, int elementType, params DimensionSpec[] dimensions)
        {
            using var valueInfo = new MemoryStream();
            WriteStringField(valueInfo, 1, name);

            using var typeProto = new MemoryStream();
            using var tensorType = new MemoryStream();
            WriteVarintField(tensorType, 1, (ulong)elementType);

            using var shape = new MemoryStream();
            foreach (var dimension in dimensions)
            {
                using var dim = new MemoryStream();
                if (dimension.DimValue.HasValue)
                    WriteVarintField(dim, 1, (ulong)dimension.DimValue.Value);
                if (dimension.DimParam != null)
                    WriteStringField(dim, 2, dimension.DimParam);
                WriteMessageField(shape, 1, dim.ToArray());
            }

            WriteMessageField(tensorType, 2, shape.ToArray());
            WriteMessageField(typeProto, 1, tensorType.ToArray());
            WriteMessageField(valueInfo, 2, typeProto.ToArray());
            graph.WriteMessageField(11, valueInfo.ToArray());
        }

        static void WriteInitializer(GraphWriter graph, string name, int rawDataBytes = 0)
        {
            using var tensor = new MemoryStream();
            var localForbiddenReadRanges = new List<ByteRange>();
            WriteStringField(tensor, 8, name);
            if (rawDataBytes > 0)
                WriteForbiddenBytesField(tensor, 9, rawDataBytes, localForbiddenReadRanges);

            graph.WriteMessageField(5, tensor.ToArray(), localForbiddenReadRanges);
        }

        static void WriteLargeNodePayload(GraphWriter graph, int payloadBytes)
        {
            graph.WriteForbiddenBytesField(1, payloadBytes);
        }

        static DimensionSpec DimValue(long value) => new DimensionSpec(value, null);
        static DimensionSpec DimParam(string value) => new DimensionSpec(null, value);
        static DimensionSpec UnknownDim() => new DimensionSpec(null, null);

        static void WriteStringField(Stream stream, int field, string value)
        {
            var bytes = Encoding.UTF8.GetBytes(value);
            WriteTag(stream, field, 2);
            WriteVarint(stream, (ulong)bytes.Length);
            stream.Write(bytes, 0, bytes.Length);
        }

        static void WriteVarintField(Stream stream, int field, ulong value)
        {
            WriteTag(stream, field, 0);
            WriteVarint(stream, value);
        }

        static void WriteMessageField(Stream stream, int field, byte[] message)
        {
            WriteMessageField(stream, field, message, null, null);
        }

        static void WriteMessageField(
            Stream stream,
            int field,
            byte[] message,
            List<ByteRange> sourceForbiddenReadRanges,
            List<ByteRange> targetForbiddenReadRanges)
        {
            WriteTag(stream, field, 2);
            WriteVarint(stream, (ulong)message.Length);
            long contentStart = stream.Position;
            stream.Write(message, 0, message.Length);

            if (sourceForbiddenReadRanges == null || targetForbiddenReadRanges == null)
                return;

            foreach (var range in sourceForbiddenReadRanges)
                targetForbiddenReadRanges.Add(new ByteRange(contentStart + range.Start, contentStart + range.End));
        }

        static void WriteForbiddenBytesField(
            Stream stream,
            int field,
            int byteCount,
            List<ByteRange> forbiddenReadRanges)
        {
            WriteTag(stream, field, 2);
            WriteVarint(stream, (ulong)byteCount);
            long start = stream.Position;
            WriteRepeatedByte(stream, byteCount);
            forbiddenReadRanges.Add(new ByteRange(start, start + byteCount));
        }

        static void WriteRepeatedByte(Stream stream, int byteCount)
        {
            var buffer = new byte[Math.Min(byteCount, 4096)];
            for (int remaining = byteCount; remaining > 0;)
            {
                int count = Math.Min(remaining, buffer.Length);
                stream.Write(buffer, 0, count);
                remaining -= count;
            }
        }

        static void WriteTag(Stream stream, int field, int wireType)
        {
            WriteVarint(stream, (ulong)((field << 3) | wireType));
        }

        static void WriteVarint(Stream stream, ulong value)
        {
            while (value >= 0x80)
            {
                stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            stream.WriteByte((byte)value);
        }

        readonly struct DimensionSpec
        {
            public readonly long? DimValue;
            public readonly string DimParam;

            public DimensionSpec(long? dimValue, string dimParam)
            {
                DimValue = dimValue;
                DimParam = dimParam;
            }
        }

        readonly struct BuiltModel
        {
            public readonly byte[] Bytes;
            public readonly List<ByteRange> ForbiddenReadRanges;

            public BuiltModel(byte[] bytes, List<ByteRange> forbiddenReadRanges)
            {
                Bytes = bytes;
                ForbiddenReadRanges = forbiddenReadRanges;
            }
        }

        readonly struct ByteRange
        {
            public readonly long Start;
            public readonly long End;

            public ByteRange(long start, long end)
            {
                Start = start;
                End = end;
            }

            public bool Intersects(long start, long end)
            {
                return start < End && end > Start;
            }
        }

        sealed class GraphWriter
        {
            readonly MemoryStream m_Stream = new MemoryStream();
            readonly List<ByteRange> m_ForbiddenReadRanges = new List<ByteRange>();

            public byte[] Bytes => m_Stream.ToArray();
            public List<ByteRange> ForbiddenReadRanges => m_ForbiddenReadRanges;

            public void WriteMessageField(int field, byte[] message, List<ByteRange> forbiddenReadRanges = null)
            {
                OnnxModelSignatureReaderTests.WriteMessageField(
                    m_Stream,
                    field,
                    message,
                    forbiddenReadRanges,
                    m_ForbiddenReadRanges);
            }

            public void WriteForbiddenBytesField(int field, int byteCount)
            {
                OnnxModelSignatureReaderTests.WriteForbiddenBytesField(
                    m_Stream,
                    field,
                    byteCount,
                    m_ForbiddenReadRanges);
            }
        }

        sealed class ThrowOnReadRangeStream : Stream
        {
            readonly byte[] m_Bytes;
            readonly List<ByteRange> m_ForbiddenReadRanges;
            long m_Position;

            public ThrowOnReadRangeStream(byte[] bytes, List<ByteRange> forbiddenReadRanges)
            {
                m_Bytes = bytes;
                m_ForbiddenReadRanges = forbiddenReadRanges;
            }

            public override bool CanRead => true;
            public override bool CanSeek => true;
            public override bool CanWrite => false;
            public override long Length => m_Bytes.Length;

            public override long Position
            {
                get => m_Position;
                set => m_Position = value;
            }

            public override int Read(byte[] buffer, int offset, int count)
            {
                if (count <= 0)
                    return 0;

                long end = Math.Min(m_Position + count, Length);
                foreach (var range in m_ForbiddenReadRanges)
                {
                    if (range.Intersects(m_Position, end))
                        throw new InvalidOperationException("Reader attempted to read a payload that should have been skipped.");
                }

                int read = (int)(end - m_Position);
                Array.Copy(m_Bytes, m_Position, buffer, offset, read);
                m_Position = end;
                return read;
            }

            public override int ReadByte()
            {
                foreach (var range in m_ForbiddenReadRanges)
                {
                    if (range.Intersects(m_Position, m_Position + 1))
                        throw new InvalidOperationException("Reader attempted to read a payload that should have been skipped.");
                }

                if (m_Position >= Length)
                    return -1;

                return m_Bytes[(int)m_Position++];
            }

            public override long Seek(long offset, SeekOrigin origin)
            {
                m_Position = origin switch
                {
                    SeekOrigin.Begin => offset,
                    SeekOrigin.Current => m_Position + offset,
                    SeekOrigin.End => Length + offset,
                    _ => throw new ArgumentOutOfRangeException(nameof(origin), origin, null)
                };
                return m_Position;
            }

            public override void Flush() { }

            public override void SetLength(long value)
            {
                throw new NotSupportedException();
            }

            public override void Write(byte[] buffer, int offset, int count)
            {
                throw new NotSupportedException();
            }
        }
    }
}
