import java.util.ArrayList;

public class PerceptronMargin extends Perceptron {
	
	public PerceptronMargin(int n) {
		super(n);
	}

	// Method: buildClassifier
	public void buildClassifier(ArrayList<Integer> data, double eta) {

		int numAttributes = data.size() - 1;
		int labelIndex = data.size() - 1;
		//System.out.print(numAttributes);

		// compute wx+theta
		double computedY = 0;
		for (int i = 0; i < numAttributes; i++) {
			computedY += w[i] * data.get(i);
		}
		computedY = computedY + theta;

		// true or false
		int trueLabel = data.get(labelIndex);
		if (computedY * trueLabel <= 1) {
			w = updateWeights(w, trueLabel, data, eta);
			theta = updateTheta(theta, trueLabel, data, eta);
		}

	}

	// Method: update w
	public double[] updateWeights(double[] w, int trueLabel,
			ArrayList<Integer> data, double eta) {

		for (int i = 0; i < w.length; i++) {
			w[i] = w[i] + eta * trueLabel * (double) data.get(i);
		}

		return w;
	}

	// Method: update theta
	public double updateTheta(double theta, int trueLabel,
			ArrayList<Integer> data, double eta) {

		theta = theta + eta * trueLabel;
		return theta;
	}

}
