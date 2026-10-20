// Synced from tools.doji.formats.onnx by tools/sync-onnx-parser/sync.sh. Do not edit.
// Source: tools.doji.formats.onnx/Runtime/ProtoReader.cs, top-level classes made internal.
// source-sha256: d5c1f6343cc426967b0d159dc49f0ffc97735d4d663a63b2f1db34d38975b144
using System;
using System.IO;
using System.Text;

namespace Doji.ONNX
{
    /// <summary>
    /// Reads protobuf wire-format primitives from a stream.
    /// </summary>
    /// <remarks>
    /// Methods that take a <c>limit</c> use <see cref="Stream.Position"/> and need a seekable
    /// stream; <see cref="EnsureSeekable"/> provides one.
    /// Methods that read a length-delimited value take the absolute stream position where
    /// the enclosing message ends. They reject a length that runs past that end, so a
    /// corrupt length can neither read into the next field nor allocate a huge buffer.
    /// </remarks>
    internal static class ProtoReader
    {
        const int MaxVarintBytes = 10;
        const int SkipBufferSize = 4096;

        /// <summary>Reads a field tag. A tag with field number 0 is invalid.</summary>
        public static uint ReadTag(Stream stream)
        {
            uint tag = ReadVarint32(stream);
            if ((tag >> 3) == 0)
                throw new InvalidDataException("Invalid protobuf tag: field number 0.");
            return tag;
        }

        /// <summary>
        /// Reads a varint and keeps its low 32 bits. Accepts the 10-byte form that protobuf
        /// writes for negative int32 and enum values.
        /// </summary>
        public static uint ReadVarint32(Stream stream) => (uint)ReadVarint64(stream);

        /// <summary>Reads a varint of up to 10 bytes.</summary>
        public static ulong ReadVarint64(Stream stream)
        {
            ulong result = 0;
            for (int i = 0; i < MaxVarintBytes; i++)
            {
                int b = stream.ReadByte();
                if (b < 0)
                    throw new EndOfStreamException();

                result |= (ulong)(b & 0x7F) << (7 * i);
                if ((b & 0x80) == 0)
                    return result;
            }

            throw new InvalidDataException("Malformed varint: more than 10 bytes.");
        }

        /// <summary>Reads an <c>int32</c> or enum value.</summary>
        public static int ReadInt32(Stream stream) => (int)ReadVarint64(stream);

        /// <summary>Reads an <c>int64</c> value.</summary>
        public static long ReadInt64(Stream stream) => (long)ReadVarint64(stream);

        /// <summary>Reads a <c>bool</c> value.</summary>
        public static bool ReadBool(Stream stream) => ReadVarint64(stream) != 0;

        public static int DecodeZigZag32(uint n) => (int)(n >> 1) ^ -(int)(n & 1);
        public static long DecodeZigZag64(ulong n) => (long)(n >> 1) ^ -(long)(n & 1);

        /// <summary>Reads a little-endian <c>fixed32</c> value.</summary>
        public static uint ReadFixed32(Stream stream)
        {
            Span<byte> buffer = stackalloc byte[4];
            ReadExactly(stream, buffer);
            return buffer[0]
                   | ((uint)buffer[1] << 8)
                   | ((uint)buffer[2] << 16)
                   | ((uint)buffer[3] << 24);
        }

        /// <summary>Reads a little-endian <c>fixed64</c> value.</summary>
        public static ulong ReadFixed64(Stream stream)
        {
            Span<byte> buffer = stackalloc byte[8];
            ReadExactly(stream, buffer);
            ulong low = buffer[0]
                        | ((uint)buffer[1] << 8)
                        | ((uint)buffer[2] << 16)
                        | ((uint)buffer[3] << 24);
            ulong high = buffer[4]
                         | ((uint)buffer[5] << 8)
                         | ((uint)buffer[6] << 16)
                         | ((uint)buffer[7] << 24);
            return low | (high << 32);
        }

        public static float ReadFloat(Stream stream) => BitConverter.Int32BitsToSingle((int)ReadFixed32(stream));
        public static double ReadDouble(Stream stream) => BitConverter.Int64BitsToDouble((long)ReadFixed64(stream));

        /// <summary>
        /// Reads the length prefix of a length-delimited field and returns the absolute
        /// position where the field ends.
        /// </summary>
        /// <param name="limit">The absolute position where the enclosing message ends.</param>
        public static long ReadLengthDelimitedEnd(Stream stream, long limit)
        {
            ulong length = ReadVarint64(stream);
            long remaining = limit - stream.Position;
            if (remaining < 0 || length > (ulong)remaining)
                throw new InvalidDataException(
                    $"Length-delimited field of {length} bytes runs past the end of its message.");
            return stream.Position + (long)length;
        }

        /// <summary>Reads a length-delimited byte array.</summary>
        /// <param name="limit">The absolute position where the enclosing message ends.</param>
        public static byte[] ReadBytes(Stream stream, long limit)
        {
            long end = ReadLengthDelimitedEnd(stream, limit);
            long length = end - stream.Position;
            if (length > int.MaxValue)
                throw new InvalidDataException($"Bytes field of {length} bytes is too large.");

            var buffer = new byte[length];
            ReadExactly(stream, buffer);
            return buffer;
        }

        /// <summary>Reads a length-delimited UTF-8 string.</summary>
        /// <param name="limit">The absolute position where the enclosing message ends.</param>
        public static string ReadString(Stream stream, long limit) => Encoding.UTF8.GetString(ReadBytes(stream, limit));

        /// <summary>Skips one field value of the given wire type.</summary>
        /// <param name="limit">The absolute position where the enclosing message ends.</param>
        public static void SkipField(Stream stream, int wireType, long limit)
        {
            switch (wireType)
            {
                case 0:
                    ReadVarint64(stream);
                    break;
                case 1:
                    SkipTo(stream, CheckedEnd(stream, 8, limit));
                    break;
                case 2:
                    SkipTo(stream, ReadLengthDelimitedEnd(stream, limit));
                    break;
                case 5:
                    SkipTo(stream, CheckedEnd(stream, 4, limit));
                    break;
                default:
                    throw new InvalidDataException($"Unsupported protobuf wire type: {wireType}.");
            }
        }

        /// <summary>
        /// Returns the number of fixed-size values in a packed field that ends at
        /// <paramref name="packedEnd"/>.
        /// </summary>
        public static int PackedFixedCount(Stream stream, long packedEnd, int valueSize)
        {
            long length = packedEnd - stream.Position;
            if (length % valueSize != 0)
                throw new InvalidDataException(
                    $"Packed field of {length} bytes is not a multiple of the {valueSize}-byte value size.");
            if (length / valueSize > int.MaxValue)
                throw new InvalidDataException($"Packed field of {length} bytes is too large.");
            return (int)(length / valueSize);
        }

        /// <summary>Throws when a message or packed field did not end exactly at <paramref name="end"/>.</summary>
        public static void ExpectEnd(Stream stream, long end)
        {
            if (stream.Position != end)
                throw new InvalidDataException(
                    $"Protobuf value ended at position {stream.Position}, expected {end}.");
        }

        /// <summary>
        /// Returns a seekable stream with the same remaining content. Copies a
        /// non-seekable stream into memory.
        /// </summary>
        public static Stream EnsureSeekable(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (stream.CanSeek)
                return stream;

            var copy = new MemoryStream();
            stream.CopyTo(copy);
            copy.Position = 0;
            return copy;
        }

        static long CheckedEnd(Stream stream, int count, long limit)
        {
            if (limit - stream.Position < count)
                throw new InvalidDataException("Fixed-size field runs past the end of its message.");
            return stream.Position + count;
        }

        static void SkipTo(Stream stream, long end)
        {
            // Setting Position past the end succeeds silently, so check first.
            if (end > stream.Length)
                throw new EndOfStreamException();
            stream.Position = end;
        }

        static void ReadExactly(Stream stream, Span<byte> buffer)
        {
            while (buffer.Length > 0)
            {
                int read = stream.Read(buffer);
                if (read <= 0)
                    throw new EndOfStreamException();
                buffer = buffer.Slice(read);
            }
        }
    }

    internal static class ProtoBytesExtensions
    {
        public static string ToStringUtf8(this byte[] bytes)
        {
            return bytes == null ? null : Encoding.UTF8.GetString(bytes);
        }
    }
}
