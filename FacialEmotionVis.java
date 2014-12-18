import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.Arrays;
import java.util.*;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.security.SecureRandom;

public class FacialEmotionVis {

    public static int NUM_ATTRIBUTES = 7;
    public static int NUM_TEST_IMAGES = 200;
    public static long seed = 1;
    public static boolean debug = false;
    public static String execCommand = null;
    public static String testingFile = null;
    public static String trainingFile = null;
    public static String folder = "";
    public SecureRandom rnd;
    public static int W,H;

    // Dataset mean
    public static double mean[] = {0.0607296,0.0594753,0.0605205,0.535278,0.222327,0.0272813,0.034389};

    public void printMessage(String s) {
        if (debug) {
            System.out.println(s);
        }
    }

    public int[] imageToArray(String imageFile) throws Exception {
        printMessage("Reading image from " + folder + imageFile);
        BufferedImage img = ImageIO.read(new File(folder + imageFile));

        H = img.getHeight();
        W = img.getWidth();
        int[] res = new int[H * W];

        int pos = 0;
        byte[] pixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        for (int i = 0; i < pixels.length; i+=3) {
            int v0 = (int)(pixels[i]);
            if (v0<0) v0 += 256;
            int v1 = (int)(pixels[i+1]);
            if (v1<0) v1 += 256;
            int v2 = (int)(pixels[i+2]);
            if (v2<0) v2 += 256;
            res[pos++] = v0 | (v1<<8) | (v2<<16);
        }
        return res;
    }

    class Face {
        int[] imgData;
        double[] attributes = new double[NUM_ATTRIBUTES];

        public void Load(String fileName, double[] attr) throws Exception {
            this.imgData = imageToArray(fileName);
            for (int i=0;i<NUM_ATTRIBUTES;i++)
            {
                this.attributes[i] = attr[i];
            }
        }
    }

    public double doExec() throws Exception {

        try {
            rnd = SecureRandom.getInstance("SHA1PRNG");
        } catch (Exception e) {
            System.err.println("ERROR: unable to generate test case.");
            System.exit(1);
        }
        rnd.setSeed(seed);

        // launch solution
        printMessage("Executing your solution: " + execCommand);
        Process solution = Runtime.getRuntime().exec(execCommand);

        BufferedReader reader = new BufferedReader(new InputStreamReader(solution.getInputStream()));
        PrintWriter writer = new PrintWriter(solution.getOutputStream());
        new ErrorStreamRedirector(solution.getErrorStream()).start();

        Face aFace = new Face();
        if (trainingFile != null) {
            BufferedReader br = new BufferedReader(new FileReader(trainingFile));
            int N = Integer.parseInt(br.readLine());
            printMessage("Training with " + N + " images");

            writer.println(N);
            writer.flush();
            for (int i=0;i<N;i++)
            {
                String s = br.readLine();
                String[] items = s.split(",");
                double[] attr = new double[NUM_ATTRIBUTES];
                for (int j=0;j<NUM_ATTRIBUTES;j++)
                    attr[j] = Double.parseDouble(items[j+1]);
                aFace.Load(items[0], attr);

                // Call training func
                writer.println(aFace.imgData.length);
                for (int v : aFace.imgData) {
                    writer.println(v);
                }
                writer.flush();
                for (double v : aFace.attributes) {
                    writer.println(v);
                }
                writer.flush();
                int ret = Integer.parseInt(reader.readLine());
                if (ret==1) break; // stop receiving training images
            }
            br.close();
        } else
        {
            System.out.println("ERROR: Training file not provided");
            System.exit(0);
        }

        double score = 0;
        double SSE = 0;
        double SSEbase = 0;
        double[] userAns = new double[NUM_ATTRIBUTES];
        double[] sqdist = new double[NUM_ATTRIBUTES];
        for (int i=0;i<NUM_ATTRIBUTES;i++) sqdist[i] = 0;
        if (testingFile != null) {
            BufferedReader br = new BufferedReader(new FileReader(testingFile));
            int N = Integer.parseInt(br.readLine());

            int[] testImg = new int[N];
            for (int i=0;i<N;i++) testImg[i] = 0;
            for (int i=0;i<NUM_TEST_IMAGES;i++)
            {
                int idx = 0;
                do
                {
                    idx = rnd.nextInt(N);
                } while (testImg[idx]==1);
                testImg[idx] = 1;
            }

            printMessage("Testing with " + NUM_TEST_IMAGES + " images");

            writer.println(NUM_TEST_IMAGES);
            writer.flush();
            for (int i=0;i<N;i++)
            {
                String s = br.readLine();
                if (testImg[i]==0) continue;
                String[] items = s.split(",");
                double[] modelAns = new double[NUM_ATTRIBUTES];
                for (int j=0;j<NUM_ATTRIBUTES;j++)
                    modelAns[j] = Double.parseDouble(items[j+1]);
                aFace.Load(items[0], modelAns);

                // Call testing func
                writer.println(aFace.imgData.length);
                for (int v : aFace.imgData) {
                    writer.println(v);
                }
                writer.flush();

                for (int j=0;j<NUM_ATTRIBUTES;j++)
                {
                    double ret = Double.parseDouble(reader.readLine());
                    userAns[j] = ret;
                }
                for (int j=0;j<NUM_ATTRIBUTES;j++)
                {
                    double E = (userAns[j] - modelAns[j])*(userAns[j] - modelAns[j]);
                    sqdist[j] += E;
                    SSE += E;
                    SSEbase += (mean[j] - modelAns[j])*(mean[j] - modelAns[j]);
                }
            }
            System.out.println("SSE = " +SSE);
            System.out.println("SSE Base = " +SSEbase);
            System.out.println("SSE on [angry] = "+sqdist[0]);
            System.out.println("SSE on [anxious] = "+sqdist[1]);
            System.out.println("SSE on [confident] = "+sqdist[2]);
            System.out.println("SSE on [happy] = "+sqdist[3]);
            System.out.println("SSE on [neutral] = "+sqdist[4]);
            System.out.println("SSE on [sad] = "+sqdist[5]);
            System.out.println("SSE on [surprised] = "+sqdist[6]);

            score = 1000000.0 * Math.max(1.0 - SSE/SSEbase, 0.0);
            br.close();
        } else
        {
            System.out.println("ERROR: Testing file not provided");
            System.exit(0);
        }
        return score;
    }

    public static void main(String[] args) throws Exception {

       for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-folder")) {
                folder = args[++i];
            } else if (args[i].equals("-exec")) {
                execCommand = args[++i];
            } else if (args[i].equals("-seed")) {
                seed = Long.parseLong(args[++i]);
            } else if (args[i].equals("-train")) {
                trainingFile = args[++i];
            } else if (args[i].equals("-test")) {
                testingFile = args[++i];
            } else if (args[i].equals("-silent")) {
                debug = false;
            } else {
                System.out.println("WARNING: unknown argument " + args[i] + ".");
            }

        }

        FacialEmotionVis vis = new FacialEmotionVis();
        try {
            double score = vis.doExec();
            System.out.println("Score  = " + score);
        } catch (Exception e) {
            System.out.println("FAILURE: " + e.getMessage());
            e.printStackTrace();
        }
    }

    class ErrorStreamRedirector extends Thread {
        public BufferedReader reader;

        public ErrorStreamRedirector(InputStream is) {
            reader = new BufferedReader(new InputStreamReader(is));
        }

        public void run() {
            while (true) {
                String s;
                try {
                    s = reader.readLine();
                } catch (Exception e) {
                    // e.printStackTrace();
                    return;
                }
                if (s == null) {
                    break;
                }
                System.out.println(s);
            }
        }
    }
}
