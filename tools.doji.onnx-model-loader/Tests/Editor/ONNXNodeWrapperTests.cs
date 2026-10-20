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

        // GetOptionalFloatArray looks for type FLOAT, not FLOATS, exactly like the Sentis
        // importer it follows, so it is not used here.
        [Test]
        public void GetRequiredFloatArray_FloatsAttribute_ReturnsValues()
        {
            var floats = new ProtoWriter().RawFloat(0.5f).RawFloat(2f);
            var attribute = new ProtoWriter()
                .String(1, "scales")
                .Int64(20, ProtoWriter.AttributeTypeFloats)
                .Message(7, floats);
            var node = ParseNode(new ProtoWriter().String(4, "Resize").Message(5, attribute));

            Assert.That(node.GetRequiredFloatArray("scales"), Is.EqualTo(new[] { 0.5f, 2f }));
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
