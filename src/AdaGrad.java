import java.util.ArrayList;

public class AdaGrad {

	double[] w;
	double theta;
	double[] g;
	double[] G;

	public AdaGrad(int n) {
		w = new double[n];
		theta = 0;
		g = new double[n + 1];
		G = new double[n + 1];

	}

	public void buildClassifier(ArrayList<Integer> data, double eta) {

		int numAttributes = data.size() - 1;
		int labelIndex = data.size() - 1;

		// compute wx+theta
		double computedY = 0;
		for (int i = 0; i < w.length; i++) {
			computedY += w[i] * data.get(i);
		}
		computedY = computedY + theta;

		// true or false
		int trueLabel = data.get(labelIndex);
		if (computedY * trueLabel <= 1) {
			computeGradientVector(data, trueLabel); // w & theta
			computeGradientSum();
			updateWeights(eta, trueLabel, data);
			updateTheta(eta, trueLabel, data);
			//System.out.println(theta);
		}
	}

	public void computeGradientVector(ArrayList<Integer> data,
			int trueLabel) {

		for (int i = 0; i < g.length - 1; i++) {
			g[i] = -trueLabel * (double)data.get(i);
		} // w
		g[g.length - 1] = -trueLabel; // theta
	}

	public void computeGradientSum() {

		for (int i = 0; i < G.length; i++) {
			G[i] = G[i] + Math.pow(g[i], 2.0);
		}

	}

	public void updateWeights(
			double eta, int trueLabel, ArrayList<Integer> data) {
		for (int i = 0; i < g.length - 1; i++) {
			if (G[i] > 0) {
				w[i] = w[i] + eta * (double)trueLabel * (double)data.get(i)
						/ Math.sqrt(G[i]);
			}
		}
	}

	public void updateTheta(double eta,
			int trueLabel, ArrayList<Integer> data) {

		if (G[g.length - 1] > 0) {
			theta = theta + eta * (double)trueLabel
					/ Math.sqrt(G[g.length - 1]);
		}

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
			return 1;
		} 
		return -1;

	}

}
