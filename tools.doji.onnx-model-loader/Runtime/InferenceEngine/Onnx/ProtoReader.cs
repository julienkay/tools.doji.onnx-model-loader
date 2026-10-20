using System;
using System.IO;
using System.Text;

namespace Doji.ONNX {

    public static class ProtoReader {
        public static uint ReadVarint32(Stream stream) {
            int shift = 0;
            uint result = 0;

            while (shift < 32) {
                int b = stream.ReadByte();
                if (b < 0)
                    throw new EndOfStreamException();

                result |= (uint)(b & 0x7F) << shift;
                if ((b & 0x80) == 0)
                    return result;

                shift += 7;
            }

            throw new InvalidDataException("Malformed Varint32");
        }

        public static ulong ReadVarint64(Stream stream) {
            int shift = 0;
            ulong result = 0;

            while (shift < 64) {
                int b = stream.ReadByte();
                if (b < 0)
                    throw new EndOfStreamException();

                result |= ((ulong)b & 0x7FUL) << shift;
                if ((b & 0x80) == 0)
                    return result;

                shift += 7;
            }

            throw new InvalidDataException("Malformed Varint64");
        }

        public static int DecodeZigZag32(uint n) => (int)((n >> 1) ^ (~(n & 1) + 1));
        public static long DecodeZigZag64(ulong n) => (long)((n >> 1) ^ (~(n & 1) + 1));

        public static string ReadString(Stream stream) {
            uint length = ReadVarint32(stream);
            var buffer = new byte[length];
            int read = stream.Read(buffer, 0, (int)length);
            if (read != length)
                throw new EndOfStreamException();
            return Encoding.UTF8.GetString(buffer);
        }

        public static byte[] ReadBytes(Stream stream) {
            uint length = ReadVarint32(stream);
            var buffer = new byte[length];
            int read = stream.Read(buffer, 0, (int)length);
            if (read != length)
                throw new EndOfStreamException();
            return buffer;
        }

        public static void SkipBytes(Stream stream) {
            uint length = ReadVarint32(stream);
            stream.Seek(length, SeekOrigin.Current);
        }

        public static void SkipField(Stream stream, int wireType) {
            switch (wireType) {
                case 0: // varint
                    ReadVarint64(stream);
                    break;
                case 1: // fixed64
                    stream.Seek(8, SeekOrigin.Current);
                    break;
                case 2: // length-delimited
                    SkipBytes(stream);
                    break;
                case 5: // fixed32
                    stream.Seek(4, SeekOrigin.Current);
                    break;
                default:
                    throw new InvalidDataException("Unknown wire type: " + wireType);
            }
        }

        public static float ReadFloat(Stream stream) {
            var buffer = new byte[4];
            stream.Read(buffer, 0, 4);
            return BitConverter.ToSingle(buffer, 0);
        }

        public static double ReadDouble(Stream stream) {
            var buffer = new byte[8];
            stream.Read(buffer, 0, 8);
            return BitConverter.ToDouble(buffer, 0);
        }
    }

    public static class ProtoBytesExtensions {
        public static string ToStringUtf8(this byte[] bytes) {
            return bytes == null ? null : Encoding.UTF8.GetString(bytes);
        }
    }
}
