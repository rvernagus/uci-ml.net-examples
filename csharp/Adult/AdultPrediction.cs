using Microsoft.ML.Data;

namespace Adult
{
    public class AdultPrediction
    {
        [ColumnName("Label")]
        public bool ActualTarget { get; set; }

        public float Probability { get; set; }

        [ColumnName("PredictedLabel")]
        public bool PredictedTarget { get; set; }

        public override string ToString() =>
            $"{{ ActualTarget: {ActualTarget}\n  Probability: {Probability}\n  PredictedTarget: {PredictedTarget} }}";
    }
}
