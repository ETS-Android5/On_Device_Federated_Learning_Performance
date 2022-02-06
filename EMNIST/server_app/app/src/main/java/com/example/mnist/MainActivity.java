package com.example.mnist;

import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.Iterator;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

import android.util.Log;

import java.net.Socket;
import java.util.Set;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import au.com.bytecode.opencsv.CSVWriter;


public class MainActivity extends AppCompatActivity {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/server";
    private static final String mnistTesturl = "https://github.com/KANG-FU/On_device_FL_dataset/raw/main/mnist_test.tar.gz";

    TextView text;
    double targetAccuracy = 0.95;
    int trainNum = 0 ;
    String trainDataset = "mnist_iid";
            
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.button);
        text = (TextView) findViewById(R.id.textView2);

        File serverDir = new File(basePath);
        if (!serverDir.exists()) {
            serverDir.mkdirs();
        }


        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute("");
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });

    }

    private class AsyncTaskRunner extends AsyncTask<String, Integer, String> {
        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            text.setText("Generate the initial model and send to clients...");
        }

        // This is our main background thread for training the model and uploading the model
        @Override
        protected String doInBackground(String... params) {
            final int numClient = 2;
            int numConnectedClient = 0;
            double accuracy = 0;
            double precision = 0;
            double recall = 0;
            double f1_score = 0;
            int num = 1;

            try {
                // Downloading the dataset from the Internet
                if (!new File(basePath + "/mnist_test").exists()) {
                    Log.d("Data download", "Data downloaded from " + mnistTesturl);
                    File modelDir = new File(basePath + "/mnist_test");
                    if (!modelDir.exists()) {
                        modelDir.mkdirs();
                    }
                    if (DataUtilities.downloadFile(mnistTesturl, basePath)) {
                        DataUtilities.extractTarGz(basePath+"/mnist_test.tar.gz", basePath + "/mnist_test");
                    }
                }

                // set up the server socket and set the port as 5000
                ServerSocket ss = new ServerSocket(5000);
                System.out.println("ServerSocket awaiting connections...");

                // define the thread for receiving the message from multiple clients
                Thread[] thread = new Thread[numClient];
                Thread[] initModelThread = new Thread[numClient];
                Thread[] signalThread = new Thread[numClient];

                modelBuildAndTrainAndEval model = new modelBuildAndTrainAndEval();
                // generate the initial model
                model.modelGenrated();

                // write the evaluation output to csv
                File file = new File(basePath + "/"+ trainDataset + ".csv");
                FileWriter output = null;
                try {
                    output = new FileWriter(file);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                CSVWriter write = new CSVWriter(output);
                // Header column value
                String[] header = { "ID", "Accuracy", "Precision", "Recall", "F1 Score" };
                write.writeNext(header);

                // The training process
                // First the server sends the model to all the clients then waiting for the trained model from the clients
                // Once receiving the model from all the clients, do the aggregation and evaluate the aggregated model
                // Evaluate the model to see if it reaches the target accuracy
                while (true){
                    while(numConnectedClient < numClient ){
                        Socket socketInit = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                        System.out.println("Connection from " + socketInit + "!");
                        numConnectedClient = numConnectedClient + 1;
                        initModelThread[numConnectedClient-1] = new Thread(new ServiceSend(socketInit));
                        initModelThread[numConnectedClient-1].start();
                    }
                    numConnectedClient = 0;
                    Thread.sleep(1000);
                    System.out.println("Waiting the trained model from clients...");
                    while(numConnectedClient < numClient ){
                        Socket socket = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                        System.out.println("Connection from " + socket + "!");
                        numConnectedClient = numConnectedClient + 1;
                        thread[numConnectedClient-1] = new Thread(new ServiceReceive(socket));
                        thread[numConnectedClient-1].start();
                        Thread.sleep(5000);
                    }
                    String updatedModelPath_cID_1 = basePath  + "/localModel_cID_1.zip";
                    String updatedModelPath_cID_2 = basePath  + "/localModel_cID_2.zip";
                    while(new File(updatedModelPath_cID_1).length() < 300000 || new File(updatedModelPath_cID_2).length() < 300000){
                        Thread.sleep(2000);
                    }
                    Thread.sleep(5000);
                    
                    double startTime = System.nanoTime();
                    MultiLayerNetwork modelLoad_cID_1 = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath_cID_1);
                    MultiLayerNetwork modelLoad_cID_2 = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath_cID_2);
                    weightAveraging(modelLoad_cID_1,modelLoad_cID_2);
                    String updatedModelPath = basePath + "/updatedModel.zip";
                    MultiLayerNetwork updatedModel = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath);
                    Evaluation eval = model.modelEval(updatedModel);
                    double aggragateAndEvalTime = (System.nanoTime()-startTime)/1000000000;
                    System.out.println("The time for evalation is "+ aggragateAndEvalTime + " s");

                    accuracy = eval.accuracy();
                    precision = eval.precision();
                    recall = eval.recall();
                    f1_score = eval.f1();
                    String[] data={String.valueOf(num),String.valueOf(accuracy),String.valueOf(precision),String.valueOf(recall),String.valueOf(f1_score)};
                    write.writeNext(data);
                    num = num + 1;

                    numConnectedClient = 0;
                    thread[0].interrupt();
                    thread[1].interrupt();
                    trainNum = trainNum + 1;
                    if (accuracy >= targetAccuracy){
                        System.out.println("Save the metrics output");
                        write.close();
                        while(numConnectedClient < numClient ){
                            Socket socket = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                            System.out.println("Connection from " + socket + "!");
                            numConnectedClient = numConnectedClient + 1;
                            signalThread[numConnectedClient-1] = new Thread(new stopSignalSend(socket));
                            signalThread[numConnectedClient-1].start();
                        }
                        break;

                    }else{
                        while(numConnectedClient < numClient ){
                            Socket socket = ss.accept(); // blocking call, this will wait until a connection is attempted on this port.
                            System.out.println("Connection from " + socket + "!");
                            numConnectedClient = numConnectedClient + 1;
                            signalThread[numConnectedClient-1] = new Thread(new ContinueSignalSend(socket));
                            signalThread[numConnectedClient-1].start();
                        }
                        numConnectedClient = 0;
                    }

                }
            }catch(Exception e){
                e.printStackTrace();
            }
            return "";
        }

        //This is called from background thread but runs in UI for a progress indicator
        @Override
        protected void onProgressUpdate(Integer... values) {

            super.onProgressUpdate(values);
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            text.setText("The accuracy reach the target "+ targetAccuracy + "\n" + "It has taken " +  trainNum + "times of training iteration" );

        }

        // The function for the weight averaging
        private void weightAveraging(MultiLayerNetwork modelLoad_cID_1,MultiLayerNetwork modelLoad_cID_2) throws IOException {
            INDArray meanCoefficient;
            meanCoefficient = modelLoad_cID_1.params().addi(modelLoad_cID_2.params()).divi(2);
            INDArray meanUpdaterState;
            meanUpdaterState = modelLoad_cID_1.updaterState().addi(modelLoad_cID_2.updaterState()).divi(2);
            MultiLayerNetwork updatedModel = new MultiLayerNetwork(modelLoad_cID_1.getLayerWiseConfigurations(), meanCoefficient);
            updatedModel.getUpdater().setStateViewArray(updatedModel, meanUpdaterState, false);
            ModelSerializer.writeModel(updatedModel, new File(basePath + "/updatedModel.zip"), true);
            System.out.println("Model has been updated in the server");
        }
    }
    // The class for building the model and evaluate the model
    // It has two methods. One is to generate the initial model. The other is to do the aggregation model evaluation.
    private  class modelBuildAndTrainAndEval {
        final int numRows = 28;
        final int numColumns = 28;
        int channels = 1; // single channel for grayscale images
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1234; // random number seed for reproducibility
        Random randNumGen = new Random(rngSeed);

        private void modelGenrated() throws IOException {

            Map<Integer, Double> learningRateSchedule = new HashMap<>();
            learningRateSchedule.put(0, 0.06);
            learningRateSchedule.put(200, 0.05);
            learningRateSchedule.put(400, 0.028);
            learningRateSchedule.put(600, 0.0060);
            learningRateSchedule.put(800, 0.001);

            ConvolutionLayer conv1 = new ConvolutionLayer.Builder(5, 5)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(20)
                    .activation(Activation.IDENTITY)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                    .gradientNormalizationThreshold(1)
                    .build();

            SubsamplingLayer maxpool1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            ConvolutionLayer conv2 = new ConvolutionLayer.Builder(5, 5)
                    .stride(1, 1) // nIn need not specified in later layers
                    .nOut(50)
                    .activation(Activation.IDENTITY)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
//                    .gradientNormalizationThreshold(1)
                    .build();

            SubsamplingLayer maxpool2 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            DenseLayer fc = new DenseLayer.Builder().activation(Activation.RELU)
                    .nOut(500)
                    .build();

            OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .build();

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(rngSeed)
                    .l2(0.0005) // ridge regression value
                    .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule)))
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .setInputType(InputType.convolutionalFlat(numRows, numColumns, channels)) // InputType.convolutional for normal image
                    .layer(conv1)
                    .layer(maxpool1)
                    .layer(conv2)
                    .layer(maxpool2)
                    .layer(fc)
                    .layer(outputLayer)
                    .build();


            MultiLayerNetwork myNetwork = new MultiLayerNetwork(conf);
            myNetwork.init();
            myNetwork.setListeners(new ScoreIterationListener(10));
            ModelSerializer.writeModel(myNetwork,  new File(basePath + "/updatedModel.zip"), false);
            System.out.println("The initial model has been generated!");
        }


        private Evaluation modelEval(MultiLayerNetwork myNetwork) throws IOException, InterruptedException {

//            // vectorization of test data
            File testData = new File(basePath + "/emnist_test");
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
            FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            ImageRecordReader testRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
            testRR.initialize(testSplit);
            DataSetIterator mnistTest = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            mnistTest.setPreProcessor(scaler);

            Log.d("evaluate model", "Evaluate model....");
            Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
            while(mnistTest.hasNext()){
                DataSet next = mnistTest.next();
                INDArray output = myNetwork.output(next.getFeatures()); //get the networks prediction
                eval.eval(next.getLabels(), output); //check the prediction against the true class
            }

            //Since we used global variables to store the classification results, no need to return
            //a results string. If the results were returned here they would be passed to onPostExecute.
            Log.d("evaluate stats", eval.stats());
            Log.d("finished","****************Example finished********************");
           return eval;
        }

    }

    // The class for multithreads of receiving service
    class ServiceReceive implements Runnable{
        private DataInputStream din;
        private DataOutputStream dout;
        public Socket socket;
        public ServiceReceive(Socket socket){
            this.socket = socket;
        }

        @Override
        public void run() {
            try{
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                receiveFile();
                Thread.sleep(10000000);
            } catch (IOException | InterruptedException e) {
                try {
                    System.out.println("interrupt the thread... ");
                    dout.writeUTF("FINISH");
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            }
        }
        // receive the files from the clients
        private void receiveFile() {
            int bytesRead = 0, current = 0;

            try {
                String name = din.readUTF();
                int fileLength = din.readInt();
                byte[] byteArray = new byte[fileLength];					//creating byteArray with length same as file length
                BufferedInputStream bis = new BufferedInputStream(din);
                File file = new File( basePath + "/" + name );
                //fileFoundFlag is a Flag which denotes the file is present or absent from the Server directory, is present int 0 is sent, else 1
                int fileFoundFlag = din.readInt();
                //System.out.println(fileFoundFlag);
                if(fileFoundFlag == 1)
                    return;
                BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
                bytesRead = bis.read(byteArray, 0, byteArray.length);			//reads bytes of length byteArray from BufferedInputStream & writes into the byteArray, (Offset 0 and length is of byteArray)
                current = bytesRead;
                //Sometimes only a portion of the file is read, hence to read the remaining portion...
                do {
                    //BufferedInputStream is read again into the byteArray, offset is current (which is the amount of bytes read previously) and length is the empty space in the byteArray after current is subtracted from its length
                    bytesRead = bis.read(byteArray, current, (byteArray.length - current));
                    if(bytesRead >= 0)
                        current += bytesRead;					//current is updated after the new bytes are read
                } while(bytesRead > 0);
                bos.write(byteArray, 0, current);				//writes bytes from the byteArray into the BufferedOutputStream, offset is 0 and length is current (which is the amount of bytes read into byteArray)
                bos.close();
                System.out.println("        File "  + " Successfully Downloaded!" );
                dout.writeInt(0);						//writeInt is used to reset if any bytes are present in the buffer after the file transfer
            }
            catch(IOException ex) {
                ex.printStackTrace();
            }
        }
    }
    // The class for multithreads of sending service
    class ServiceSend implements Runnable{
        public DataInputStream din;
        public DataOutputStream dout;
        public Socket socket;
        public ServiceSend(Socket socket){
            this.socket = socket;
        }

        @Override
        public void run() {
            try{
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                System.out.println("model sending ...");
                File initModel = new File( basePath + "/updatedModel.zip");
                sendFile(initModel);
                System.out.println("model sent");
            } catch (IOException e) {
                e.printStackTrace();
            }

        }
        // send the files to clients
        public void sendFile(File file) {
            try {
                byte[] byteArray = new byte[(int) file.length()];					//creating byteArray with length same as file length
                dout.writeInt(byteArray.length);
                BufferedInputStream bis = new BufferedInputStream (new FileInputStream(file));
                //Writing int 0 as a Flag which denotes the file is present in the Server directory, if file was absent, FileNotFound exception will be thrown and int 1 will be written
                dout.writeInt(0);
                BufferedOutputStream bos = new BufferedOutputStream(dout);

                int count;
                while((count = bis.read(byteArray)) != -1) {			//reads bytes of byteArray length from the BufferedInputStream into byteArray
                    bos.write(byteArray, 0, count);					//writes bytes from byteArray into the BufferedOutputStream (0 is the offset and count is the length)
                }
                bos.flush();
                bis.close();
                din.readInt();//readInt is used to reset if any bytes are present in the buffer after the file transfer
            }
            catch(FileNotFoundException ex) {
                System.out.println("File Not Found!");
                try {
                    //Writing int 1 as a Flag which denotes the file is absent from the Server directory, if file was present int 0 would be written
                    dout.writeInt(1);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            catch(IOException ex) {
                ex.printStackTrace();
            }

        }
    }
    // The class for multithreads of sending continuing signal to clients so that clients can continue train the model
    class ContinueSignalSend implements Runnable{
        public DataInputStream din;
        public DataOutputStream dout;
        public Socket socket;
        public ContinueSignalSend(Socket socket){
            this.socket = socket;
        }

        @Override
        public void run() {
            try{
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                System.out.println("Send the signal to inform client of continuing training or not...");
                dout.writeInt(1);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }
    // The class for multithreads of sending stop signal to clients so that clients do not continue train the model
    class stopSignalSend implements Runnable{
        public DataInputStream din;
        public DataOutputStream dout;
        public Socket socket;
        public stopSignalSend(Socket socket){
            this.socket = socket;
        }

        @Override
        public void run() {
            try{
                din = new DataInputStream(socket.getInputStream());
                dout = new DataOutputStream(socket.getOutputStream());
                System.out.println("Send the signal to inform client of continuing training or not...");
                dout.writeInt(0);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

    }

}

