namespace Annealing
{
    public class AnnealPrediction
    {
        public string LabelValue { get; set; }

        public float[] Score { get; set; }

        public string PredictedLabelValue { get; set; }

        public override string ToString() =>
            $"{{ LabelValue: {LabelValue}\n  Score: {string.Join(",", Score)}\n  PredictedLabelValue: {PredictedLabelValue} }}";
    }
}
