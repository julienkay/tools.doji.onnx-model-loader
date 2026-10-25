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
        public const int AttributeTypeString = 3;
        public const int AttributeTypeStrings = 8;
        public const int DataTypeFloat = 1;

        /// <summary>An AttributeProto of type INTS, encoded as an exporter writes it.</summary>
        public static ProtoWriter IntsAttribute(string name, params long[] values)
        {
            return new ProtoWriter().String(1, name).Int64(20, AttributeTypeInts).PackedInt64(8, values);
        }

        /// <summary>An AttributeProto of type STRING.</summary>
        public static ProtoWriter StringAttribute(string name, string value)
        {
            return new ProtoWriter().String(1, name).Int64(20, AttributeTypeString).String(4, value);
        }

        /// <summary>A float TensorProto whose data is in an external file at <paramref name="location"/>.</summary>
        public static ProtoWriter ExternalFloatTensor(string name, string location, params long[] dims)
        {
            var tensor = new ProtoWriter();
            foreach (var dim in dims)
                tensor.Int64(1, dim);
            return tensor
                .Int64(2, DataTypeFloat)
                .String(8, name)
                .Message(13, new ProtoWriter().String(1, "location").String(2, location))
                .Message(13, new ProtoWriter().String(1, "offset").String(2, "0"))
                .Message(13, new ProtoWriter().String(1, "length").String(2, "4"))
                .Int64(14, 1); // data_location = EXTERNAL
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
