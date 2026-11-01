using System;
using System.IO;
using Doji.ONNX;
using Unity.InferenceEngine;
using Unity.Jobs;
using OnnxDataType = Doji.ONNX.TensorProto.Types.DataType;

namespace Doji.AI.InferenceEngine
{
    /// <summary>
    /// Builds Sentis constants from tensor data that <see cref="OnnxModelData"/> left in the file buffer.
    /// </summary>
    static class ONNXConstantsLoader
    {
        /// <summary>
        /// Loads the data of <paramref name="tensorProto"/>, an initializer or a Constant node value of
        /// <paramref name="data"/>. With <paramref name="borrow"/>, float32 and int32 raw_data reaches the
        /// constant as a segment of the file buffer, without a copy. Every other encoding is converted
        /// into one new array. External data is read from files in <paramref name="directoryPath"/>.
        /// </summary>
        public static ConstantTensor LoadConstant(OnnxModelData data, TensorProto tensorProto, string directoryPath, bool borrow)
        {
            if (!data.TryGetData(tensorProto, out var tensor))
                throw new OnnxLayerImportException($"Tensor {tensorProto.Name} is not part of the loaded model.");

            var shape = GetShape(tensorProto);
            var dataType = (OnnxDataType)tensorProto.DataType;

            if (tensor.IsExternal)
            {
                // Read straight into the final array when no conversion follows.
                var range = tensor.GetExternalRange();
                long length = range.Length ?? (long)shape.length * ElementSize(dataType);
                var bytes = new byte[checked((int)length)];
                tensor.ReadExternalTo(directoryPath, bytes);
                return FromBytes(shape, dataType, bytes, keep: true);
            }

            if (tensor.TryGetRawBytes(out var raw))
                return FromBytes(shape, dataType, raw, keep: borrow);

            // Typed fields (float_data, int64_data, ...): DecodeTo writes the raw_data layout.
            var decoded = new byte[checked((int)tensor.DecodedByteLength)];
            tensor.DecodeTo(decoded);
            return FromBytes(shape, dataType, decoded, keep: true);
        }

        /// <summary>
        /// Converts bytes in the raw_data layout. With <paramref name="keep"/>, float32 and int32 constants
        /// use <paramref name="bytes"/> as they are; without it, they get a copy.
        /// </summary>
        static ConstantTensor FromBytes(TensorShape shape, OnnxDataType dataType, ArraySegment<byte> bytes, bool keep)
        {
            ReadOnlySpan<byte> src = bytes;
            switch (dataType)
            {
                case OnnxDataType.Float:
                    CheckLength(src, sizeof(float), shape, dataType);
                    return new ConstantTensor(shape, DataType.Float, keep ? bytes : src.ToArray());
                case OnnxDataType.Int32:
                    CheckLength(src, sizeof(int), shape, dataType);
                    return new ConstantTensor(shape, DataType.Int, keep ? bytes : src.ToArray());
                case OnnxDataType.Double:
                    return FloatFromFloat64(shape, src);
                case OnnxDataType.Float16:
                    return FloatFromFloat16(shape, src);
                case OnnxDataType.Int64:
                    return IntFromInt64(shape, src);
                case OnnxDataType.Bool:
                    return IntFromBool(shape, src);
                case OnnxDataType.Uint8:
                    return IntFromUint8(shape, src);
                case OnnxDataType.Int8:
                    return IntFromInt8(shape, src);
                default:
                    throw new OnnxLayerImportException($"Tensor data type {dataType} is not supported.");
            }
        }

        static int ElementSize(OnnxDataType dataType) => dataType switch
        {
            OnnxDataType.Double or OnnxDataType.Int64 => 8,
            OnnxDataType.Float or OnnxDataType.Int32 => 4,
            OnnxDataType.Float16 => 2,
            OnnxDataType.Bool or OnnxDataType.Uint8 or OnnxDataType.Int8 => 1,
            _ => throw new OnnxLayerImportException($"Tensor data type {dataType} is not supported.")
        };

        static void CheckLength(ReadOnlySpan<byte> src, int elementSize, TensorShape shape, OnnxDataType dataType)
        {
            if (src.Length != (long)elementSize * shape.length)
                throw new OnnxLayerImportException($"{dataType} tensor of shape {shape} has {src.Length} bytes of data, expected {(long)elementSize * shape.length}.");
        }

        // Span versions of the ConstantTensor conversions, so borrowed bytes convert without a copy.
        // Source offsets are not aligned to the element size; the jobs read unaligned values.

        static unsafe ConstantTensor FloatFromFloat16(TensorShape shape, ReadOnlySpan<byte> src)
        {
            CheckLength(src, sizeof(ushort), shape, OnnxDataType.Float16);
            var dst = new byte[sizeof(float) * shape.length];
            if (shape.length > 0)
                fixed (byte* s = src, d = dst)
                    new BurstJobsCastTensor.Float16BytesAsFloatJob { src = (ushort*)s, dst = (float*)d }.Schedule(shape.length, 32).Complete();
            return new ConstantTensor(shape, DataType.Float, dst);
        }

        static unsafe ConstantTensor FloatFromFloat64(TensorShape shape, ReadOnlySpan<byte> src)
        {
            CheckLength(src, sizeof(double), shape, OnnxDataType.Double);
            var dst = new byte[sizeof(float) * shape.length];
            if (shape.length > 0)
                fixed (byte* s = src, d = dst)
                    new BurstJobsCastTensor.Float64BytesAsFloatJob { src = (long*)s, dst = (float*)d }.Schedule(shape.length, 32).Complete();
            return new ConstantTensor(shape, DataType.Float, dst);
        }

        static unsafe ConstantTensor IntFromInt64(TensorShape shape, ReadOnlySpan<byte> src)
        {
            CheckLength(src, sizeof(long), shape, OnnxDataType.Int64);
            var dst = new byte[sizeof(int) * shape.length];
            if (shape.length > 0)
                fixed (byte* s = src, d = dst)
                    new BurstJobsCastTensor.Int64BytesAsIntJob { src = (long*)s, dst = (int*)d }.Schedule(shape.length, 32).Complete();
            return new ConstantTensor(shape, DataType.Int, dst);
        }

        static unsafe ConstantTensor IntFromBool(TensorShape shape, ReadOnlySpan<byte> src)
        {
            CheckLength(src, sizeof(bool), shape, OnnxDataType.Bool);
            var dst = new byte[sizeof(int) * shape.length];
            if (shape.length > 0)
                fixed (byte* s = src, d = dst)
                    new BurstJobsCastTensor.BoolBytesAsIntJob { src = (bool*)s, dst = (int*)d }.Schedule(shape.length, 32).Complete();
            return new ConstantTensor(shape, DataType.Int, dst);
        }

        static unsafe ConstantTensor IntFromUint8(TensorShape shape, ReadOnlySpan<byte> src)
        {
            CheckLength(src, sizeof(byte), shape, OnnxDataType.Uint8);
            var dst = new byte[sizeof(int) * shape.length];
            if (shape.length > 0)
                fixed (byte* s = src, d = dst)
                    new BurstJobsCastTensor.Uint8BytesAsIntJob { src = s, dst = (int*)d }.Schedule(shape.length, 32).Complete();
            return new ConstantTensor(shape, DataType.Int, dst);
        }

        static unsafe ConstantTensor IntFromInt8(TensorShape shape, ReadOnlySpan<byte> src)
        {
            CheckLength(src, sizeof(sbyte), shape, OnnxDataType.Int8);
            var dst = new byte[sizeof(int) * shape.length];
            if (shape.length > 0)
                fixed (byte* s = src, d = dst)
                    new BurstJobsCastTensor.Int8BytesAsIntJob { src = (sbyte*)s, dst = (int*)d }.Schedule(shape.length, 32).Complete();
            return new ConstantTensor(shape, DataType.Int, dst);
        }

        static TensorShape GetShape(TensorProto tensorProto)
        {
            var onnxShape = new int[tensorProto.Dims.Count];
            for (var i = 0; i < onnxShape.Length; i++)
            {
                var v = tensorProto.Dims[i];
                onnxShape[i] = v < int.MinValue ? int.MinValue : v > int.MaxValue ? int.MaxValue : (int)v;
            }

            return new TensorShape(onnxShape);
        }
    }
}
