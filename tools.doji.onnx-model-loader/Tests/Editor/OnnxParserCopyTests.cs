using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NUnit.Framework;
using Unity.InferenceEngine;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    /// <summary>
    /// The loader compiles internal copies of the tools.doji.formats.onnx parser into
    /// Unity.InferenceEngine. These tests fail when a copy is out of date or public.
    /// </summary>
    public class OnnxParserCopyTests
    {
        const string CopyDirectory = "Packages/tools.doji.onnx-model-loader/Runtime/InferenceEngine/Onnx";
        const string SourceDirectory = "Packages/tools.doji.formats.onnx/Runtime";
        const string HashPrefix = "// source-sha256: ";

        [TestCase("Onnx.cs")]
        [TestCase("ProtoReader.cs")]
        public void Copy_MatchesFormatsOnnxSource(string fileName)
        {
            var sourcePath = Path.Combine(SourceDirectory, fileName);
            var copyPath = Path.Combine(CopyDirectory, fileName);
            Assert.That(File.Exists(sourcePath), Is.True, $"Missing {sourcePath}");
            Assert.That(File.Exists(copyPath), Is.True, $"Missing {copyPath}");

            var stampLine = File.ReadLines(copyPath).FirstOrDefault(line => line.StartsWith(HashPrefix));
            Assert.That(stampLine, Is.Not.Null, $"{copyPath} has no '{HashPrefix.Trim()}' line.");

            Assert.That(stampLine.Substring(HashPrefix.Length).Trim(), Is.EqualTo(Sha256WithoutCarriageReturns(sourcePath)),
                $"{copyPath} is out of date. Run tools/sync-onnx-parser/sync.sh.");
        }

        [Test]
        public void Copies_ExposeNoPublicTypes()
        {
            // Public copies would clash (CS0433) with tools.doji.formats.onnx in any assembly
            // that references both it and Unity.InferenceEngine.
            var publicCopies = typeof(Model).Assembly.GetExportedTypes()
                .Where(type => type.Namespace == "Doji.ONNX")
                .Select(type => type.FullName)
                .ToArray();

            Assert.That(publicCopies, Is.Empty);
        }

        /// <summary>Line-ending independent, so a CRLF checkout hashes like the LF source.</summary>
        static string Sha256WithoutCarriageReturns(string path)
        {
            var bytes = File.ReadAllBytes(path).Where(b => b != (byte)'\r').ToArray();
            using var sha = SHA256.Create();
            return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }
    }
}
