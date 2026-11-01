using Doji.AI.InferenceEngine;
using NUnit.Framework;
using Act = Unity.InferenceEngine.Layers.RnnActivation;

namespace Doji.AI.ONNXModelLoader.Editor.Tests
{
    /// <summary>
    /// LSTM <c>activation_alpha</c> and <c>activation_beta</c> values, assigned as ONNX Runtime
    /// assigns them: in order, only to activations that use them.
    /// </summary>
    public class LstmActivationArgumentsTests
    {
        [Test]
        public void NoValues_UsesOperatorDefaults()
        {
            var activations = new[] { Act.HardSigmoid, Act.LeakyRelu, Act.Tanh };

            ONNXModelConverter.GetLstmActivationArguments(activations, null, null, out var alpha, out var beta);

            Assert.That(alpha, Is.EqualTo(new[] { 0.2f, 0.01f, 0f }));
            Assert.That(beta, Is.EqualTo(new[] { 0.5f, 0f, 0f }));
        }

        [Test]
        public void Values_SkipActivationsWithoutArguments()
        {
            // Sigmoid and Tanh take no alpha, so 0.3 belongs to LeakyRelu.
            var activations = new[] { Act.Sigmoid, Act.LeakyRelu, Act.Tanh };

            ONNXModelConverter.GetLstmActivationArguments(activations, new[] { 0.3f }, null, out var alpha, out _);

            Assert.That(alpha, Is.EqualTo(new[] { 0f, 0.3f, 0f }));
        }

        [Test]
        public void Values_FewerThanActivations_RestUseDefaults()
        {
            var activations = new[] { Act.HardSigmoid, Act.ScaledTanh, Act.ScaledTanh };

            ONNXModelConverter.GetLstmActivationArguments(activations, new[] { 0.1666f }, new[] { 0.4f, 2f }, out var alpha, out var beta);

            Assert.That(alpha, Is.EqualTo(new[] { 0.1666f, 1f, 1f }));
            Assert.That(beta, Is.EqualTo(new[] { 0.4f, 2f, 1f }));
        }

        [Test]
        public void BetaValues_ConsumedOnlyByActivationsWithBeta()
        {
            // LeakyRelu takes alpha but no beta, so the first beta belongs to Affine.
            var activations = new[] { Act.LeakyRelu, Act.Affine, Act.Tanh };

            ONNXModelConverter.GetLstmActivationArguments(activations, new[] { 0.2f, 3f }, new[] { 0.5f }, out var alpha, out var beta);

            Assert.That(alpha, Is.EqualTo(new[] { 0.2f, 3f, 0f }));
            Assert.That(beta, Is.EqualTo(new[] { 0f, 0.5f, 0f }));
        }
    }
}
