import java.util.ArrayList;

/*
 * Changes from Perceptron: function used for updating weight (take in alpha) & fixed theta=#Samples & iniliaze w=1
 * 
 */

public class Winnow {

	double[] w;
	double theta;
	
	private Winnow() {
		
	}
	
	public Winnow(int n) {
		w = new double[n];
		for (int i = 0; i < n; i++) {
			w[i] = 1;
		}
		theta = -n;
	}

	// Method: buildClassifier
	public void buildClassifier(ArrayList<Integer> data, double alpha, int n) {

		int numAttributes = data.size() - 1; // also, the number of elements in
												// w
		int labelIndex = data.size() - 1;

		// compute wx+theta
		double computedY = 0;
		for (int i = 0; i < numAttributes; i++) {
			computedY += w[i] * data.get(i);
		}
		computedY = computedY + theta;

		// true or false
		int trueLabel = data.get(labelIndex);
		if (computedY * trueLabel <= 0) {
			w = updateWeights(w, trueLabel, data, alpha);
		}

	}

	// Method: update w
	public double[] updateWeights(double[] w, int trueLabel,
			ArrayList<Integer> data, double alpha) {

		for (int i = 0; i < w.length; i++) {
			w[i] = w[i] * Math.pow(alpha, trueLabel * data.get(i));
		}
		return w;
	}

	// Method: classify sample
	public int classifyData(ArrayList<Integer> sample) {

		int numAttributes = sample.size() - 1;
		double y = 0;

		for (int i = 0; i < numAttributes; i++) {
			y += w[i] * sample.get(i);
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
