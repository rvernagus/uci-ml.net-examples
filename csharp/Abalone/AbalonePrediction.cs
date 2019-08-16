using System;

namespace Abalone
{
    public class AbalonePrediction
    {
        public Single Label { get; set; }

        public Single Score { get; set; }

        public override string ToString() =>
            $"Label: {Label}, Score: {Score}";
    }
}
