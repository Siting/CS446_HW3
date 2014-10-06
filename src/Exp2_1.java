import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;
import java.util.StringTokenizer;

public class Exp2_1 {

	public static void main(String[] args) throws IOException {

		// Load file
		if (args.length != 1) {
			System.out.println("need path");
			return;
		}
		File file = new File(args[0]);
		Scanner s = null;
		try {
			s = new Scanner(file);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		// Save file to dataMatrix
		ArrayList<ArrayList<Integer>> dataMatrix = new ArrayList<ArrayList<Integer>>();
		while (s.hasNextLine()) {
			String line = s.nextLine();
			StringTokenizer tokenizer = new StringTokenizer(line, ",");
			ArrayList<Integer> dataRow = new ArrayList<Integer>();
			while (tokenizer.hasMoreElements()) {
				dataRow.add(Integer.valueOf(tokenizer.nextToken()));
			}
			dataMatrix.add(dataRow);
		}
		s.close();

		// Generate training & test data
		int numSamples = dataMatrix.size();
		// int numAttributes = dataMatrix.get(0).size();
		int numTenPercept = (int) Math.floor((double) 0.1 * numSamples);

		ArrayList<ArrayList<Integer>> trainingMatrix = new ArrayList<ArrayList<Integer>>();
		for (int i = 0; i < numTenPercept; i++) {
			trainingMatrix.add(dataMatrix.get(i));
		}

		ArrayList<ArrayList<Integer>> testMatrix = new ArrayList<ArrayList<Integer>>();
		for (int i = numTenPercept; i < 2 * numTenPercept; i++) {
			testMatrix.add(dataMatrix.get(i));
			// System.out.println(dataMatrix.get(i).get(500));
		}

		// Iterate through algorithms
		int n = 40;
		double[] etaCandiPercepM = { 1.5, 0.25, 0.03, 0.005, 0.001 };
		double[] alphaCandiWinnow = { 1.1, 1.01, 1.005, 1.0005, 1.0001 };
		double[] alphaCandiWinnowM = { 1.1, 1.01, 1.005, 1.0005, 1.0001 };
		double[] gammaCandiWinnowM = { 2.0, 0.3, 0.04, 0.006, 0.001 };
		double[] etaCandiAdaGrad = { 1.5, 0.25, 0.03, 0.005, 0.001 };

		// Tuning
//		 ParaTuning paraTune = new ParaTuning();
//		 paraTune.tunePercepM(trainingMatrix, testMatrix, etaCandiPercepM, n);
//		 paraTune.tuneWinnow(trainingMatrix, testMatrix, alphaCandiWinnow, n);
//		 paraTune.tuneWinnowM(trainingMatrix, testMatrix, alphaCandiWinnowM,
//		 gammaCandiWinnowM, n);
//		 paraTune.tuneAdaGrad(trainingMatrix, testMatrix, etaCandiAdaGrad, n);

		// Training on the entire dataset
		double etaPercepM = 1.5;
		double alphaWinnow = 1.1;
		double alphaWinnowM = 1.1;
		double gammaWinnowM = 2.0;
		double etaAdaGrad = 0.25;

		Perceptron percep = new Perceptron(n);
		PerceptronMargin percepM = new PerceptronMargin(n);
		Winnow winnow = new Winnow(n);
		WinnowMargin winnowM = new WinnowMargin(n);
		AdaGrad adaGrad = new AdaGrad(n);

		ArrayList<ArrayList<Integer>> cumuError = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> errorInitial = new ArrayList<Integer>();
		errorInitial.add(0);
		errorInitial.add(0);
		cumuError.add(errorInitial);

		for (int algo = 0; algo < 5; algo++) {
			int cumuE = 0;
			int correct = 0;
			for (int i = 0; i < dataMatrix.size(); i++) {

				ArrayList<Integer> data = dataMatrix.get(i);
				int pred = 0;

				if (algo == 0) {
					pred = percep.classifyData(data);
					percep.buildClassifier(data);		
				} else if (algo == 1) {
					pred = percepM.classifyData(data);
					percepM.buildClassifier(data, etaPercepM);				
				} else if (algo == 2) {
					pred = winnow.classifyData(data);
					winnow.buildClassifier(data, alphaWinnow, n);				
				} else if (algo == 3) {
					pred = winnowM.classifyData(data);
					winnowM.buildClassifier(data, alphaWinnowM, gammaWinnowM, n);					
				} else if (algo == 4) {
					pred = adaGrad.classifyData(data);
					adaGrad.buildClassifier(data, etaAdaGrad);				
				}

				if (pred != data.get(data.size() - 1)) {
					cumuE = cumuE + 1;

				} else {
					correct++;
				}
				
				if ((i+1)%1 == 0) {
					ArrayList<Integer> error = new ArrayList<Integer>();
					error.add(i + 1);
					error.add(cumuE);
					cumuError.add(error);
				}
				
				if (correct == 1000) {
					System.out.println("Algorithm converged");
					break;
				}

				// if ( i%25000 == 0) {
				// System.out.println("Finished feeding sample " + i);
				// }

			}

			// print to file
			PrintWriter out = new PrintWriter("exp2_1_output_algo" + algo
					+ ".txt");
			for (int i = 0; i < cumuError.size(); i++) {
				out.println(cumuError.get(i).get(0) + ","
						+ cumuError.get(i).get(1));
			}
			out.close();

			System.out.println("Algorithm " + algo + " has " + cumuE
					+ " mistakes");

		}

	}
}
