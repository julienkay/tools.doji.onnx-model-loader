using System;
using System.Collections.Generic;
using System.IO;
using SentisFlatBuffer;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Google.FlatBuffers;

namespace Doji.AI.InferenceEngine
{
    /// <summary>
    /// Writes a .sentis file as <see cref="ModelWriter.Save(string, Model)"/> does, but one weights chunk at a time,
    /// and reads the data of a constant only when its chunk is written. <see cref="ModelWriter"/> holds every chunk
    /// in memory before it writes the first one, so it needs twice the size of the weights.
    /// </summary>
    static class StreamedModelWriter
    {
        // As in ModelWriter.
        const int k_WeightsFlatBufferOverhead = 32;
        const long k_MaxConstantSize = int.MaxValue - k_WeightsFlatBufferOverhead;

        /// <summary>
        /// Saves <paramref name="model"/> to <paramref name="path"/>. A constant without data gets it from
        /// <paramref name="loadData"/> while its chunk is written, so the file holds a weight that was never in the model.
        /// </summary>
        public static void Save(string path, Model model, Func<Constant, ArraySegment<byte>> loadData)
        {
            var description = SaveDescription(model, out var chunks, out var offsets);
            using var stream = File.Create(path);
            stream.Write(description);
            foreach (var (constants, length) in chunks)
            {
                var data = new byte[length];
                foreach (var i in constants)
                {
                    var constant = model.constants[i];
                    if (constant.lengthBytes == 0)
                        continue;
                    var bytes = constant.array.Array != null ? constant.array : loadData(constant);
                    if (bytes.Count != constant.lengthBytes)
                        throw new InvalidDataException($"Constant {constant.index} has {bytes.Count} bytes, expected {constant.lengthBytes}.");
                    System.Buffer.BlockCopy(bytes.Array, bytes.Offset, data, offsets[i], bytes.Count);
                }

                // Preallocate exact size for FlatBuffer otherwise allocator can overshoot max FlatBuffer size for large byte arrays
                var builder = new FlatBufferBuilder(k_WeightsFlatBufferOverhead + data.Length);
                var storage = SentisFlatBuffer.Buffer.CreateStorageVectorBlock(builder, data);
                data = null;
                SentisFlatBuffer.Buffer.StartBuffer(builder);
                SentisFlatBuffer.Buffer.AddStorage(builder, storage);
                var cb = SentisFlatBuffer.Buffer.EndBuffer(builder);
                builder.FinishSizePrefixed(cb.Value);
                var buffer = builder.DataBuffer;
                stream.Write(buffer.ToSpan(buffer.Position, buffer.Length - buffer.Position));
                builder = null;
                GC.Collect();
            }
        }

        static VectorOffset CreateDynamicSizes(FlatBufferBuilder builder, DynamicTensorShape shape)
        {
            if (shape.isRankDynamic)
                return default;

            var dims = new Offset<EDim>[shape.rank];
            for (var i = 0; i < shape.rank; i++)
            {
                var d = shape[i];
                dims[i] = d.dimType switch
                {
                    DimType.Unknown => EDim.CreateEDim(builder, val_type: SymbolicDim.NONE),
                    DimType.Static => EDim.CreateEDim(builder, val_type: SymbolicDim.Int, Int.CreateInt(builder, d.value).Value),
                    DimType.Param => EDim.CreateEDim(builder, val_type: SymbolicDim.Byte, SentisFlatBuffer.Byte.CreateByte(builder, d.param).Value),
                    _ => throw new ArgumentOutOfRangeException()
                };
            }

            return SentisFlatBuffer.Tensor.CreateDynamicSizesVector(builder, dims);
        }

        /// <summary>
        /// The model description of ModelWriter.SaveModel (Inference Engine 2.6), and the layout of the weights:
        /// the constants and length of each chunk, and the offset of each constant in its chunk.
        /// </summary>
        static byte[] SaveDescription(Model model, out List<(List<int> constants, int length)> chunks, out int[] constantOffsets)
        {
            // infer the data types and shapes for the model before saving
            model.InferDataTypesShapes();

            var builder = new FlatBufferBuilder(1);

            // values: tensor desc + float/int layer values
            var values = new List<Offset<EValue>>();

            // mapping of indexes in model to indexes in serialized model
            var indexMapping = new Dictionary<int, int>();

            // Inputs
            var inputsIndices = new int[model.inputs.Count];
            var inputsNames = new StringOffset[model.inputs.Count];
            for (int i = 0; i < model.inputs.Count; i++)
            {
                var input = model.inputs[i];

                inputsIndices[i] = i;
                inputsNames[i] = builder.CreateString(input.name);

                Offset<SentisFlatBuffer.Tensor> val;
                if (input.shape.IsStatic())
                {
                    var tensorShape = input.shape.ToTensorShape();
                    var size = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, tensorShape.ToArray());
                    int lengthByte = tensorShape.length * AllocatorUtils.LengthInBytesPadded(tensorShape.length, input.dataType);
                    val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)input.dataType, lengthByte, size);
                }
                else
                {
                    var size = CreateDynamicSizes(builder, input.shape);
                    val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)input.dataType, shape_dynamism: TensorShapeDynamism.DYNAMIC_UNBOUND, dynamic_sizesOffset: size);
                }
                values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, val.Value));
                indexMapping[input.index] = values.Count - 1;
            }
            var epModelInputs = ExecutionPlan.CreateInputsVector(builder, inputsIndices);
            var epModelInputsNames = ExecutionPlan.CreateInputsNameVector(builder, inputsNames);

            // Model Settings
            var epModelName = builder.CreateString(model.ProducerName);

            // constants
            var constantIndexes = new Dictionary<int, int>();
            var constantBufferIndexes = new List<int>();
            constantOffsets = new int[model.constants.Count];
            chunks = new List<(List<int>, int)> { (new List<int>(), 0) };
            long weightsTotalLengthBytes = 0;
            for (var i = 0; i < model.constants.Count; i++)
            {
                var constant = model.constants[i];
                constantIndexes.Add(constant.index, i);
                if (constant.lengthBytes > k_MaxConstantSize)
                    throw new InvalidDataException($"Constant of size {constant.lengthBytes} is larger than maximum serializable constant size {k_MaxConstantSize}");
                if ((long)chunks[^1].length + constant.lengthBytes > k_MaxConstantSize)
                {
                    // constant doesn't fit in final buffer, add new one
                    chunks.Add((new List<int>(), 0));
                }

                var idx = chunks.Count - 1;
                var offset = chunks[idx].length;
                constantBufferIndexes.Add(idx);
                constantOffsets[i] = offset;
                chunks[idx].constants.Add(i);
                chunks[idx] = (chunks[idx].constants, offset + constant.lengthBytes);
                weightsTotalLengthBytes += constant.lengthBytes;
            }

            Dictionary<string, int> operatorNames = new Dictionary<string, int>();
            var operators = new List<Offset<Operator>>();
            var chains = new List<Offset<Chain>>();
            var chainCPU = new List<int>();

            for (int i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];
                if (!operatorNames.ContainsKey(layer.opName))
                {
                    var operationName = builder.CreateString(layer.opName);
                    var operation = Operator.CreateOperator(builder, operationName);
                    operators.Add(operation);
                    operatorNames[layer.opName] = operatorNames.Count;
                }

                // layer inputs
                var layerInputs = new int[layer.inputs.Length];
                for (int k = 0; k < layer.inputs.Length; k++)
                {
                    var input = layer.inputs[k];

                    if (input != -1 && !indexMapping.ContainsKey(input))
                    {
                        // layer input must be a constant
                        var constantIndex = constantIndexes[input];
                        var constant = model.constants[constantIndex];
                        var size = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, constant.shape.ToArray());
                        var val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)constant.dataType, constant.lengthBytes, size, (uint)(constantBufferIndexes[constantIndex] + 1), constantOffsets[constantIndex]);
                        values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, val.Value));
                        indexMapping[input] = values.Count - 1;
                    }

                    layerInputs[k] = (input == -1) ? -1 : indexMapping[input];
                }

                var prevValuesCount = values.Count;
                layer.SerializeFields(builder, values);
                var layerAttributesInputs = new int[values.Count - prevValuesCount];
                for (var j = 0; j < layerAttributesInputs.Length; j++)
                    layerAttributesInputs[j] = prevValuesCount + j;

                var layerOutputs = new List<int>();

                foreach (var output in layer.outputs)
                {
                    var scalarType = ScalarType.FLOAT;
                    var lengthByte = 0;
                    var fixedSizesOffset = default(VectorOffset);
                    var shapeDynamism = TensorShapeDynamism.STATIC;
                    var dynamicSizesOffset = default(VectorOffset);
                    var hasDynamicRank = false;
                    var dataType = model.GetDataType(output);
                    if (dataType.HasValue)
                        scalarType = (ScalarType)dataType.Value;
                    var shape = model.GetShape(output);
                    if (shape.HasValue)
                    {
                        shapeDynamism = shape.Value.IsStatic() ? TensorShapeDynamism.STATIC : TensorShapeDynamism.DYNAMIC_UNBOUND;
                        if (shapeDynamism == TensorShapeDynamism.STATIC)
                        {
                            var tensorShape = shape.Value.ToTensorShape();
                            fixedSizesOffset = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, tensorShape.ToArray());
                            lengthByte = tensorShape.length * AllocatorUtils.LengthInBytesPadded(tensorShape.length, (DataType)scalarType);
                        }
                        else
                        {
                            dynamicSizesOffset = CreateDynamicSizes(builder, shape.Value);
                            hasDynamicRank = shape.Value.isRankDynamic;
                        }
                    }
                    var valTo = SentisFlatBuffer.Tensor.CreateTensor(
                        builder: builder,
                        scalar_type: scalarType,
                        length_byte: lengthByte,
                        fixed_sizesOffset: fixedSizesOffset,
                        shape_dynamism: shapeDynamism,
                        dynamic_sizesOffset: dynamicSizesOffset,
                        has_dynamic_rank: hasDynamicRank
                    );
                    values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, valTo.Value));
                    layerOutputs.Add(values.Count - 1);
                    indexMapping[output] = values.Count - 1;
                }

                var layerInputVector = ExecutionPlan.CreateInputsVector(builder, layerInputs);
                var layerOutputVector = ExecutionPlan.CreateOutputsVector(builder, layerOutputs.ToArray());
                // attributes
                var layerAttributesVector = ExecutionPlan.CreateInputsVector(builder, layerAttributesInputs);
                var kernelCall = KernelCall.CreateKernelCall(builder, operatorNames[layer.opName], layerAttributesVector);
                Instruction.StartInstruction(builder);
                Instruction.AddInstrArgsType(builder, InstructionArguments.KernelCall);
                Instruction.AddInstrArgs(builder, kernelCall.Value);
                var linstruction = Instruction.EndInstruction(builder);

                var lInstructionVector = Chain.CreateInstructionsVector(builder, new[] { linstruction });
                Chain.StartChain(builder);
                Chain.AddInputs(builder, layerInputVector);
                Chain.AddOutputs(builder, layerOutputVector);
                Chain.AddInstructions(builder, lInstructionVector);
                var lChain = Chain.EndChain(builder);
                chains.Add(lChain);
            }

            for (int i = 0; i < model.outputs.Count; i++)
            {
                var output = model.outputs[i];
                if (!constantIndexes.TryGetValue(output.index, out var constantIndex))
                    continue;

                var constant = model.constants[constantIndex];

                var size = SentisFlatBuffer.Tensor.CreateFixedSizesVector(builder, constant.shape.ToArray());
                var val = SentisFlatBuffer.Tensor.CreateTensor(builder, (ScalarType)constant.dataType, constant.lengthBytes, size, (uint)(constantBufferIndexes[constantIndex] + 1), constantOffsets[constantIndex]);
                values.Add(EValue.CreateEValue(builder, KernelTypes.Tensor, val.Value));

                var constantVector = ExecutionPlan.CreateInputsVector(builder, new[] { values.Count - 1 });

                Instruction.StartInstruction(builder);
                Instruction.AddInstrArgsType(builder, InstructionArguments.NONE);
                var linstruction = Instruction.EndInstruction(builder);

                var lInstructionVector = Chain.CreateInstructionsVector(builder, new[] { linstruction });
                Chain.StartChain(builder);
                Chain.AddInputs(builder, constantVector);
                Chain.AddInstructions(builder, lInstructionVector);
                var lChain = Chain.EndChain(builder);
                chains.Add(lChain);

                indexMapping[constant.index] = values.Count - 1;
            }

            var outputIndices = new List<int>();
            var outputsNames = new List<StringOffset>();
            foreach (var output in model.outputs)
            {
                outputIndices.Add(indexMapping[output.index]);
                outputsNames.Add(builder.CreateString(output.name));
            }

            var symbolicDimNames = new StringOffset[model.symbolicDimNames?.Length ?? 0];
            for (var i = 0; i < symbolicDimNames.Length; i++)
                symbolicDimNames[i] = builder.CreateString(model.symbolicDimNames[i]);

            var epModelOutputsNames = ExecutionPlan.CreateOutputsNameVector(builder, outputsNames.ToArray());
            var epModelOutputs = ExecutionPlan.CreateOutputsVector(builder, outputIndices.ToArray());
            var epModelValues = ExecutionPlan.CreateValuesVector(builder, values.ToArray());
            var epModelOperators = ExecutionPlan.CreateOperatorsVector(builder, operators.ToArray());
            var epModelChains = ExecutionPlan.CreateChainsVector(builder, chains.ToArray());
            var epCPUChains = BackendPartitioning.CreateChainsVector(builder, chainCPU.ToArray());
            var epBackendPartitioning = BackendPartitioning.CreateBackendPartitioning(builder, epCPUChains, SentisFlatBuffer.BackendType.CPU);
            var epSymbolicDimNames = ExecutionPlan.CreateSymbolicDimNamesVector(builder, symbolicDimNames);

            ExecutionPlan.StartExecutionPlan(builder);
            ExecutionPlan.AddName(builder, epModelName);
            ExecutionPlan.AddInputs(builder, epModelInputs);
            ExecutionPlan.AddInputsName(builder, epModelInputsNames);
            ExecutionPlan.AddOutputs(builder, epModelOutputs);
            ExecutionPlan.AddOutputsName(builder, epModelOutputsNames);
            ExecutionPlan.AddValues(builder, epModelValues);
            ExecutionPlan.AddOperators(builder, epModelOperators);
            ExecutionPlan.AddChains(builder, epModelChains);
            ExecutionPlan.AddBackendPartitioning(builder, epBackendPartitioning);
            ExecutionPlan.AddSymbolicDimNames(builder, epSymbolicDimNames);
            var programExecutionPlan = ExecutionPlan.EndExecutionPlan(builder);

            var dataSegments = new Offset<DataSegment>[chunks.Count];
            for (var i = 0; i < chunks.Count; i++)
                dataSegments[i] = DataSegment.CreateDataSegment(builder, (ulong)weightsTotalLengthBytes, (ulong)chunks[i].length);
            var programDataSegments = Program.CreateSegmentsVector(builder, dataSegments);

            Program.StartProgram(builder);
            Program.AddVersion(builder, ModelWriter.version);
            Program.AddExecutionPlan(builder, programExecutionPlan);
            Program.AddSegments(builder, programDataSegments);
            var program = Program.EndProgram(builder);
            builder.FinishSizePrefixed(program.Value);

            return builder.DataBuffer.ToSizedArray();
        }
    }
}
