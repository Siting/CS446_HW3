import java.util.ArrayList;


public class Perceptron {

	double[] w;
	double theta;
	
	private Perceptron() {
		
	}
	
	public Perceptron(int n) {
		w = new double[n];
		theta = 0;
	}
	

	// Method: buildClassifier
	public void buildClassifier(ArrayList<Integer> data) {

		int numAttributes = data.size() - 1;
		int labelIndex = data.size() - 1;
		
		// compute wx+theta
		double computedY = 0;
		for (int i = 0; i < numAttributes; i++) {
			computedY += w[i] * (double) data.get(i);
		}
		computedY = computedY + theta;

		// true or false
		int trueLabel = data.get(labelIndex);
		if (computedY * trueLabel <= 0) {
			w = updateWeights(w, trueLabel, data);
			theta = updateTheta(theta, trueLabel, data);
		}

	}

	// Method: update w
	public double[] updateWeights(double[] w, int trueLabel,
			ArrayList<Integer> data) {

		for (int i = 0; i < w.length; i++) {
			w[i] = w[i] + trueLabel * data.get(i);
		}
		return w;
	}

	// Method: update theta
	public double updateTheta(double theta, int trueLabel,
			ArrayList<Integer> data) {

		theta = theta + trueLabel;
		return theta;
	}

	// Method: classify sample
	public int classifyData(ArrayList<Integer> sample) {

		int numAttributes = sample.size() - 1;
		double y = 0;

		for (int i = 0; i < numAttributes; i++) {
			y += w[i] * (double) sample.get(i);
		}
		y = y + theta;
		
		if (y >= 0) {
			y = 1;
		} else {
			y = -1;
		}

		return (int) y;
	}
}
