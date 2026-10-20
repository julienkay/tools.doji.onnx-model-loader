using System;
using System.IO;
using System.Text;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    /// <summary>Builds protobuf wire-format bytes for synthetic ONNX test messages.</summary>
    sealed class ProtoWriter
    {
        readonly MemoryStream m_Stream = new MemoryStream();

        public byte[] ToArray() => m_Stream.ToArray();

        public ProtoWriter Tag(int field, int wireType) => RawVarint((ulong)((field << 3) | wireType));

        public ProtoWriter RawVarint(ulong value)
        {
            while (value >= 0x80)
            {
                m_Stream.WriteByte((byte)(value | 0x80));
                value >>= 7;
            }

            m_Stream.WriteByte((byte)value);
            return this;
        }

        public ProtoWriter RawFloat(float value)
        {
            var bytes = BitConverter.GetBytes(value);
            m_Stream.Write(bytes, 0, bytes.Length);
            return this;
        }

        public ProtoWriter Int64(int field, long value) => Tag(field, 0).RawVarint((ulong)value);
        public ProtoWriter String(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));
        public ProtoWriter Message(int field, ProtoWriter message) => Bytes(field, message.ToArray());

        public ProtoWriter Bytes(int field, byte[] value)
        {
            Tag(field, 2).RawVarint((ulong)value.Length);
            m_Stream.Write(value, 0, value.Length);
            return this;
        }

        public ProtoWriter PackedInt64(int field, params long[] values)
        {
            var packed = new ProtoWriter();
            foreach (var value in values)
                packed.RawVarint((ulong)value);
            return Message(field, packed);
        }

        // ---- ONNX helpers ----

        public const int AttributeTypeInts = 7;
        public const int AttributeTypeFloats = 6;
        public const int AttributeTypeStrings = 8;
        public const int DataTypeFloat = 1;

        /// <summary>An AttributeProto of type INTS, encoded as an exporter writes it.</summary>
        public static ProtoWriter IntsAttribute(string name, params long[] values)
        {
            return new ProtoWriter().String(1, name).Int64(20, AttributeTypeInts).PackedInt64(8, values);
        }

        /// <summary>A ValueInfoProto for a float tensor; no shape when <paramref name="dims"/> is null.</summary>
        public static ProtoWriter FloatTensorValueInfo(string name, params long[] dims)
        {
            var tensorType = new ProtoWriter().Int64(1, DataTypeFloat);
            if (dims != null)
            {
                var shape = new ProtoWriter();
                foreach (var dim in dims)
                    shape.Message(1, new ProtoWriter().Int64(1, dim));
                tensorType.Message(2, shape);
            }

            return new ProtoWriter().String(1, name).Message(2, new ProtoWriter().Message(1, tensorType));
        }
    }
}
