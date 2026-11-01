using System.IO;
using Doji.AI.InferenceEngine;
using Doji.ONNX;
using NUnit.Framework;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    /// <summary>Attribute lookups on nodes parsed from wire bytes, as exporters write them.</summary>
    public class ONNXNodeWrapperTests
    {
        static ONNXNodeWrapper ParseNode(ProtoWriter node)
        {
            using var stream = new MemoryStream(node.ToArray());
            return new ONNXNodeWrapper(NodeProto.Deserialize(stream));
        }

        [Test]
        public void GetOptionalIntArray_IntsAttribute_ReturnsValues()
        {
            var node = ParseNode(new ProtoWriter()
                .String(4, "Conv")
                .Message(5, ProtoWriter.IntsAttribute("pads", 1, 1, 1, 1)));

            Assert.That(node.GetOptionalIntArray("pads", null), Is.EqualTo(new[] { 1, 1, 1, 1 }));
        }

        [Test]
        public void GetOptionalIntArray_NegativeInts_ReturnsValues()
        {
            var node = ParseNode(new ProtoWriter()
                .String(4, "ReduceSum")
                .Message(5, ProtoWriter.IntsAttribute("axes", -1, -2)));

            Assert.That(node.GetOptionalIntArray("axes", null), Is.EqualTo(new[] { -1, -2 }));
        }

        static ProtoWriter FloatsAttribute(string name, params float[] values)
        {
            var floats = new ProtoWriter();
            foreach (var value in values)
                floats.RawFloat(value);
            return new ProtoWriter()
                .String(1, name)
                .Int64(20, ProtoWriter.AttributeTypeFloats)
                .Message(7, floats);
        }

        [Test]
        public void GetRequiredFloatArray_FloatsAttribute_ReturnsValues()
        {
            var node = ParseNode(new ProtoWriter().String(4, "Resize").Message(5, FloatsAttribute("scales", 0.5f, 2f)));

            Assert.That(node.GetRequiredFloatArray("scales"), Is.EqualTo(new[] { 0.5f, 2f }));
        }

        [Test]
        public void GetOptionalFloatArray_FloatsAttribute_ReturnsValues()
        {
            var node = ParseNode(new ProtoWriter().String(4, "LSTM").Message(5, FloatsAttribute("activation_alpha", 0.25f, 3f)));

            Assert.That(node.GetOptionalFloatArray("activation_alpha", null), Is.EqualTo(new[] { 0.25f, 3f }));
        }

        [Test]
        public void GetOptionalFloatArray_MissingAttribute_ReturnsDefault()
        {
            var node = ParseNode(new ProtoWriter().String(4, "LSTM"));

            Assert.That(node.GetOptionalFloatArray("activation_alpha", null), Is.Null);
        }

        [Test]
        public void GetOptionalStringArray_StringsAttribute_ReturnsValues()
        {
            var attribute = new ProtoWriter()
                .String(1, "directions")
                .Int64(20, ProtoWriter.AttributeTypeStrings)
                .String(9, "forward")
                .String(9, "reverse");
            var node = ParseNode(new ProtoWriter().String(4, "Scan").Message(5, attribute));

            Assert.That(node.GetOptionalStringArray("directions", null), Is.EqualTo(new[] { "forward", "reverse" }));
        }
    }
}
