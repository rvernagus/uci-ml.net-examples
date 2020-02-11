namespace Arrhythmia
{
    public class ArrhythmiaPrediction
    {
        public int LabelValue { get; set; }

        public float[] Score { get; set; }

        public int PredictedLabelValue { get; set; }

        public override string ToString() =>
            $"{{ LabelValue: {LabelValue}\n  Score: {string.Join(",", Score)}\n  PredictedLabelValue: {PredictedLabelValue} }}";
    }
}
