package com.journaldev.androidcameraxopencv;

import static org.opencv.core.Core.FONT_HERSHEY_SIMPLEX;
import static org.opencv.core.Core.NORM_L2;
import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.Core.divide;
import static org.opencv.core.Core.magnitude;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.minMaxLoc;
import static org.opencv.core.Core.multiply;
import static org.opencv.core.Core.norm;
import static org.opencv.core.Core.normalize;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.Mat.ones;
import static org.opencv.core.Mat.zeros;
import static org.opencv.core.TermCriteria.COUNT;
import static org.opencv.core.TermCriteria.EPS;
import static org.opencv.core.TermCriteria.MAX_ITER;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.RETR_LIST;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.ellipse;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.fitEllipse;

import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static java.lang.Math.cos;
import static java.lang.Math.max;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.Exif;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.media.ExifInterface;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {


    double CV_PI = 3.1415926535897932384626433832795;
    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    TextureView textureView;
    ImageView ivBitmap;
    LinearLayout llBottom;
    Scalar COLOR_RED = new Scalar(255.0, 0.0, 0.0, 255.0);
    Scalar COLOR_GREEN = new Scalar(0.0, 255.0, 0.0, 255.0);
    Scalar COLOR_BLUE = new Scalar(0.0, 0.0, 255.0, 255.0);
    Scalar COLOR_YELLOW = new Scalar(255.0, 255.0, 0.0, 255.0);
    org.opencv.core.Size IMAGE_SIZE = new org.opencv.core.Size(1512, 2016);
    int NUMBER_OF_CCTAG_IMAGE_POINTS = 0;
    int TAKE_PHOTO = 0;
    //org.opencv.core.Size MAX_CHESSBOARD_SIZE = new org.opencv.core.Size(9,6);
    org.opencv.core.Size chessboardSize = new org.opencv.core.Size(10,8);
    org.opencv.core.Size circleGridSize = new org.opencv.core.Size(4,11);

    String currentImageProcessing = "CHESSBOARD";

    ImageCapture imageCapture;
    ImageAnalysis imageAnalysis;
    Preview preview;

    FloatingActionButton btnCapture, btnOk, btnCancel;

    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnCapture = findViewById(R.id.btnCapture);
        btnOk = findViewById(R.id.btnAccept);
        btnCancel = findViewById(R.id.btnReject);

        btnOk.setOnClickListener(this);
        btnCancel.setOnClickListener(this);

        llBottom = findViewById(R.id.llBottom);
        textureView = findViewById(R.id.textureView);
        ivBitmap = findViewById(R.id.ivBitmap);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {

        CameraX.unbindAll();

        preview = setPreview();
        imageCapture = setImageCapture();
        imageAnalysis = setImageAnalysis();

        //bind to lifecycle:
        CameraX.bindToLifecycle(this, preview, imageCapture, imageAnalysis);
    }


    private Preview setPreview() {

        Rational aspectRatio = new Rational(textureView.getWidth(), textureView.getHeight());
        Size screen = new Size(textureView.getWidth(), textureView.getHeight()); //size of the screen

        PreviewConfig pConfig = new PreviewConfig.Builder().setTargetAspectRatio(aspectRatio).setTargetResolution(screen).build();
        Preview preview = new Preview(pConfig);

        preview.setOnPreviewOutputUpdateListener(
                new Preview.OnPreviewOutputUpdateListener() {
                    @Override
                    public void onUpdated(Preview.PreviewOutput output) {
                        ViewGroup parent = (ViewGroup) textureView.getParent();
                        parent.removeView(textureView);
                        parent.addView(textureView, 0);

                        textureView.setSurfaceTexture(output.getSurfaceTexture());
                        //updateTransform();
                    }
                });

        return preview;
    }


    private ImageCapture setImageCapture() {
        ImageCaptureConfig imageCaptureConfig = new ImageCaptureConfig.Builder().setCaptureMode(ImageCapture.CaptureMode.MIN_LATENCY)
                .setTargetRotation(Surface.ROTATION_0).build();
        final ImageCapture imgCapture = new ImageCapture(imageCaptureConfig);


        btnCapture.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                if(currentImageProcessing.equals("CCTAG") && NUMBER_OF_CCTAG_IMAGE_POINTS == 47) {
                    TAKE_PHOTO = 1;
                    /*imgCapture.takePicture(new ImageCapture.OnImageCapturedListener() {
                        @Override
                        public void onCaptureSuccess(ImageProxy image, int rotationDegrees) {
                            Bitmap bitmap = textureView.getBitmap();
                            showAcceptedRejectedButton(true);
                            ivBitmap.setImageBitmap(bitmap);
                        }

                        @Override
                        public void onError(ImageCapture.UseCaseError useCaseError, String message, @Nullable Throwable cause) {
                            super.onError(useCaseError, message, cause);
                        }
                    });*/
                }
                if(currentImageProcessing.equals("ASSYMETRIC_CIRCLES") && NUMBER_OF_CCTAG_IMAGE_POINTS == 47) {
                    imgCapture.takePicture(new ImageCapture.OnImageCapturedListener() {
                        @Override
                        public void onCaptureSuccess(ImageProxy image, int rotationDegrees) {
                            Bitmap bitmap = textureView.getBitmap();
                            showAcceptedRejectedButton(true);
                            ivBitmap.setImageBitmap(bitmap);
                        }

                        @Override
                        public void onError(ImageCapture.UseCaseError useCaseError, String message, @Nullable Throwable cause) {
                            super.onError(useCaseError, message, cause);
                        }
                    });
                }


                File file = new File(
                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "" + System.currentTimeMillis() + "_JDCameraX.jpg");
                imgCapture.takePicture(file, new ImageCapture.OnImageSavedListener() {
                    @Override
                    public void onImageSaved(@NonNull File file) {
                        Bitmap bitmap = textureView.getBitmap();
                        showAcceptedRejectedButton(true);
                        ivBitmap.setImageBitmap(bitmap);
                    }

                    @Override
                    public void onError(@NonNull ImageCapture.UseCaseError useCaseError, @NonNull String message, @Nullable Throwable cause) {

                    }
                });
            }
        });

        return imgCapture;
    }


    private ImageAnalysis setImageAnalysis() {

        // Setup image analysis pipeline that computes average pixel luminance
        HandlerThread analyzerThread = new HandlerThread("OpenCVAnalysis");
        analyzerThread.start();

        ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setCallbackHandler(new Handler(analyzerThread.getLooper()))
                .setImageQueueDepth(1).build();

        ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);

        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(ImageProxy image, int rotationDegrees) {
                        //Analyzing live camera feed begins.

                        final Bitmap bitmap = textureView.getBitmap();

                        //final Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.sss);

                        if (bitmap == null)
                            return;

                        Mat matColor = new Mat();
                        Utils.bitmapToMat(bitmap, matColor);

//---------------------------------------------------------------------------
                        if (currentImageProcessing.equals("CHESSBOARD")) {
                            matColor = getChessboardCorners(bitmap);
//                            MatOfPoint2f corners = getChessboardCorners(bitmap);

//                            if(corners.toList().size() >= 4) {
//                                drawMarker(matColor, new Point(corners.get(0, 0)), COLOR_RED, 1, 10, 5, 1);
//                                drawMarker(matColor, new Point(corners.get((int) chessboardSize.width - 1, 0)), COLOR_BLUE, 1, 10, 5, 1);
//                                drawMarker(matColor, new Point(corners.get((int) (chessboardSize.width * (chessboardSize.height - 1)), 0)), COLOR_GREEN, 1, 10, 5, 1);
//                                drawMarker(matColor, new Point(corners.get((int) (chessboardSize.width * chessboardSize.height) - 1, 0)), COLOR_YELLOW, 1, 10, 5, 1);
//                            }
                        } else if (currentImageProcessing.equals("CCTAG")) {
                            matColor = detectCcTags(bitmap);
                            /*for(Point imagePoint : detectCcTags(bitmap)) {
                                circle(matColor, new Point(imagePoint.x, imagePoint.y), 3, COLOR_RED, -1);
                            }*/
                        } else if (currentImageProcessing.equals("ASSYMETRIC_CIRCLES")) {
                            Mat centers = new Mat();
                            centers = getAssymetricCircleCenters(bitmap);
                            if(centers.size().width * centers.size().height == circleGridSize.width * circleGridSize.height) {
                                drawMarker(matColor, new Point(centers.get(0, 0)[0], centers.get(0, 0)[1]), COLOR_RED, 1, 10, 5, 1);
                                drawMarker(matColor, new Point(centers.get((int) circleGridSize.width - 1, 0)[0], centers.get((int) circleGridSize.width - 1, 0)[1]), COLOR_BLUE, 1, 10, 5, 1);
                                drawMarker(matColor, new Point(centers.get((int) (circleGridSize.width * (circleGridSize.height - 1)), 0)[0], centers.get((int) (circleGridSize.width * (circleGridSize.height - 1)), 0)[1]), COLOR_GREEN, 1, 10, 5, 1);
                                drawMarker(matColor, new Point(centers.get((int) (circleGridSize.width * (circleGridSize.height) - 1), 0)[0], centers.get((int) (circleGridSize.width * (circleGridSize.height) - 1), 0)[1]), COLOR_YELLOW, 1, 10, 5, 1);
                            }
                        }
//---------------------------------------------------------------------------

                        Utils.matToBitmap(matColor, bitmap);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivBitmap.setImageBitmap(bitmap);
                            }
                        });

                    }
                });

        return imageAnalysis;
    }

    private void clearDirectory() {
        Log.d("Debug:", "Deleting Photos");

        String[] children = getFilesDir().list();

        for (String child : children) {
            File file = new File(getFilesDir() + "/" + child);
            Log.d("Debug:", String.valueOf(file));
            file.delete();
        }
    }

    private void calibration() {

        MatOfPoint3f objectPoints = new MatOfPoint3f();
        List<Point3> pointsList = new ArrayList<>();

        Mat matTest = new Mat();
        Mat matTest2 = new Mat();

        if(currentImageProcessing.equals("CCTAG")) {/*
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.reference1);
            Utils.bitmapToMat(bitmap, matTest);
            int j = 0;
            for(Point imagePoint : detectCcTags(bitmap)) {
                pointsList.add(new Point3(imagePoint.x, imagePoint.y, 0));
                circle(matTest, new Point(imagePoint.x, imagePoint.y), 3, COLOR_RED, -1);
                putText(matTest, String.valueOf(j), imagePoint,
                        FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);
                j++;
            }
            Log.d("Debug:", String.valueOf(pointsList.size()));
            objectPoints.fromList(pointsList);

            j = 0;

            File fileDir = getFilesDir();
            List<Mat> photosObjectPoints = new ArrayList<>();
            List<Mat> photosImagePoints = new ArrayList<>();
            MatOfPoint2f photoImagePoints2f = new MatOfPoint2f();

            for (int i = 0; i < fileDir.listFiles().length; i++) {
                photosObjectPoints.add(objectPoints);
                photoImagePoints2f.fromList(detectCcTags(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath())));
                Utils.bitmapToMat(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath()), matTest2);

                for(Point imagePoint : detectCcTags(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath()))) {
                    //pointsList.add(new Point3(imagePoint.x, imagePoint.y, 0));
                    circle(matTest2, new Point(imagePoint.x, imagePoint.y), 3, COLOR_RED, -1);
                    putText(matTest2, String.valueOf(j), imagePoint,
                            FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);
                    j++;
                }

                Log.d("Debug:", String.valueOf(photoImagePoints2f.size()));
                photosImagePoints.add(photoImagePoints2f);
            }

            Mat cameraMatrix = new Mat();
            MatOfDouble distCoeffs = new MatOfDouble();
            List<Mat> rvecs = new ArrayList<>();
            List<Mat> tvecs = new ArrayList<>();

            Bitmap bitmap2 = BitmapFactory.decodeFile(fileDir.listFiles()[0].getPath());
            Log.d("Debug:", String.valueOf(fileDir.listFiles()[0].getPath()));

            calibrateCamera(photosObjectPoints, photosImagePoints, IMAGE_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

            bitmap = BitmapFactory.decodeFile(fileDir.listFiles()[0].getPath());
            Mat matDistorted = new Mat();
            Mat matUndistorted = new Mat();
            Utils.bitmapToMat(bitmap, matDistorted);
            double h = matDistorted.size().height;
            double w = matDistorted.size().width;

            Mat newCameraMat = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new org.opencv.core.Size(w, h), 1);

            undistort(matDistorted, matUndistorted, cameraMatrix, distCoeffs, newCameraMat);

            double mean_error = 0;
            MatOfPoint2f imgpoints2 = new MatOfPoint2f();
            for(int i = 0; i < photosObjectPoints.size(); i++) {
                projectPoints((MatOfPoint3f) photosObjectPoints.get(i), rvecs.get(i), tvecs.get(i), cameraMatrix, distCoeffs, imgpoints2);
                double error = norm(photosImagePoints.get(i), imgpoints2, NORM_L2) / imgpoints2.toList().size();
                mean_error += error;
            }
            putText(matUndistorted, "reprojection error: " + mean_error, new Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);

            Utils.matToBitmap(matUndistorted, bitmap);
            File file = new File(getFilesDir(), "Calibrated.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {
            }*/

        } else if(currentImageProcessing.equals("ASSYMETRIC_CIRCLES")) {
            for (int y = 0; y < circleGridSize.height; ++y) {
                for (int x = 0; x < circleGridSize.width; ++x) {
                    Point3 row;
                    if(y % 2 == 0) {
                        row = new Point3(x * 2, y, 0);
                    } else {
                        row = new Point3((x * 2) + 1, y, 0);
                    }
                    pointsList.add(row);
                }
            }

            objectPoints.fromList(pointsList);

            File fileDir = getFilesDir();
            List<Mat> photosObjectPoints = new ArrayList<>();
            List<Mat> photosImagePoints = new ArrayList<>();
            Mat photoImagePoints;
            MatOfPoint2f photoImagePoints2f;

            for (int i = 0; i < fileDir.listFiles().length; i++) {
                photosObjectPoints.add(objectPoints);
                photoImagePoints = getAssymetricCircleCenters(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath()));
                photoImagePoints2f = new MatOfPoint2f(photoImagePoints);

                photosImagePoints.add(photoImagePoints2f);

                Log.d("Debug:", "" + photoImagePoints2f.size() + fileDir.listFiles()[i].getPath());
            }
            Mat cameraMatrix = new Mat();
            MatOfDouble distCoeffs = new MatOfDouble();
            List<Mat> rvecs = new ArrayList<>();
            List<Mat> tvecs = new ArrayList<>();

            calibrateCamera(photosObjectPoints, photosImagePoints, IMAGE_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

            Bitmap bitmap = BitmapFactory.decodeFile(fileDir.listFiles()[0].getPath());
            Mat matDistorted = new Mat();
            Mat matUndistorted = new Mat();
            Utils.bitmapToMat(bitmap, matDistorted);
            double h = matDistorted.size().height;
            double w = matDistorted.size().width;

            Mat newCameraMat = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new org.opencv.core.Size(w, h), 1);

            undistort(matDistorted, matUndistorted, cameraMatrix, distCoeffs, newCameraMat);

            double mean_error = 0;
            MatOfPoint2f imgpoints2 = new MatOfPoint2f();
            for(int i = 0; i < photosObjectPoints.size(); i++) {
                projectPoints((MatOfPoint3f) photosObjectPoints.get(i), rvecs.get(i), tvecs.get(i), cameraMatrix, distCoeffs, imgpoints2);
                double error = norm(photosImagePoints.get(i), imgpoints2, NORM_L2) / imgpoints2.toList().size();
                mean_error += error;
            }
            putText(matUndistorted, "reprojection error: " + mean_error, new Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);

            Utils.matToBitmap(matUndistorted, bitmap);
            File file = new File(getFilesDir(), "Calibrated.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {
            }
        }else {

            for (int y = 0; y < 6; ++y) {
                for (int x = 0; x < 9; ++x) {
                    Point3 row = new Point3(x, y, 0);
                    pointsList.add(row);
                }
            }

            objectPoints.fromList(pointsList);

            File fileDir = getFilesDir();
            List<Mat> photosObjectPoints = new ArrayList<>();
            List<Mat> photosImagePoints = new ArrayList<>();
            MatOfPoint2f photoImagePoints2f = new MatOfPoint2f();

            for (int i = 0; i < fileDir.listFiles().length; i++) {
                photosObjectPoints.add(objectPoints);
//                photoImagePoints2f = getChessboardCorners(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath()));
            /*for(int y=0; y<54; y++) {
                    Mat row = new Mat(1, 3, CV_32F);
                    row.col(0).setTo(new Scalar(photoImagePoints2f.toList().get(y).x));
                    row.col(1).setTo(new Scalar(photoImagePoints2f.toList().get(y).y));
                    row.col(2).setTo(new Scalar(0));
                    photoImagePoints.push_back(row);
            }*/
                Log.d("Debug:", "" + photoImagePoints2f.size() + fileDir.listFiles()[i].getPath());
                photosImagePoints.add(photoImagePoints2f);
            }
            Mat cameraMatrix = new Mat();
            MatOfDouble distCoeffs = new MatOfDouble();
            List<Mat> rvecs = new ArrayList<>();
            List<Mat> tvecs = new ArrayList<>();

            calibrateCamera(photosObjectPoints, photosImagePoints, IMAGE_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

            Bitmap bitmap = BitmapFactory.decodeFile(fileDir.listFiles()[0].getPath());
            Mat matDistorted = new Mat();
            Mat matUndistorted = new Mat();
            Utils.bitmapToMat(bitmap, matDistorted);
            double h = matDistorted.size().height;
            double w = matDistorted.size().width;

            Mat newCameraMat = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new org.opencv.core.Size(w, h), 1);

            undistort(matDistorted, matUndistorted, cameraMatrix, distCoeffs, newCameraMat);

            double mean_error = 0;
            MatOfPoint2f imgpoints2 = new MatOfPoint2f();
            for(int i = 0; i < photosObjectPoints.size(); i++) {
                projectPoints((MatOfPoint3f) photosObjectPoints.get(i), rvecs.get(i), tvecs.get(i), cameraMatrix, distCoeffs, imgpoints2);
                double error = norm(photosImagePoints.get(i), imgpoints2, NORM_L2) / imgpoints2.toList().size();
                mean_error += error;
            }
            putText(matUndistorted, "reprojection error: " + mean_error, new Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);

            Utils.matToBitmap(matUndistorted, bitmap);
            File file = new File(getFilesDir(), "Calibrated.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {
            }

            return;
        }
    }

    private Mat getAssymetricCircleCenters(Bitmap bitmap) {
        Mat matColor = new Mat();
        Mat matGrey = new Mat();
        Mat matGreyDownscaled = new Mat();
        Mat centers = new Mat();
        boolean result;

        Utils.bitmapToMat(bitmap, matColor);
        cvtColor(matColor, matGrey, COLOR_BGR2GRAY);

        //resize(matGrey, matGreyDownscaled, new org.opencv.core.Size(), 0.25, 0.25, INTER_NEAREST);

        result = findCirclesGrid(matGrey, circleGridSize, centers, CALIB_CB_ASYMMETRIC_GRID);

        //multiply(centers, new Scalar(4.0, 4.0), centers);

        return centers;
    }

    private Mat getChessboardCorners(Bitmap bitmap) {

        Mat matColor = new Mat();
        Mat matGrey = new Mat();
//        Mat matGreyDownscaled = new Mat();
//        MatOfPoint2f corners = new MatOfPoint2f();
//        boolean result;
        Utils.bitmapToMat(bitmap, matColor);
        cvtColor(matColor, matGrey, COLOR_BGR2GRAY);
//        resize(matGrey, matGrey, new org.opencv.core.Size(), 0.25, 0.25, INTER_NEAREST);
//        result = findChessboardCorners(matGreyDownscaled, chessboardSize, corners);
//        multiply(corners, new Scalar(4.0, 4.0), corners);
//        if(result) {
//            cornerSubPix(matGrey, corners, new org.opencv.core.Size(11, 11),
//                    new org.opencv.core.Size(-1, -1),
//                    new TermCriteria(EPS + MAX_ITER, 30, 0.001));
//        }
//        Mat gaussBlurred = new Mat();
//        GaussianBlur(matColor, gaussBlurred, new org.opencv.core.Size(5,5),0);
//        Mat maskMatrix = zeros(new org.opencv.core.Size(matGrey.width(), matGrey.height()), CV_8U);
//        Mat kernelMatrix = getStructuringElement(MORPH_ELLIPSE, new org.opencv.core.Size(11,11));
//        Mat morphedMat = new Mat(); // close
//        morphologyEx(matGrey, morphedMat, MORPH_CLOSE, kernelMatrix);
//        Mat dividedMatrix = new Mat();
//        divide(matGrey, morphedMat, dividedMatrix);
//        normalize(dividedMatrix, dividedMatrix,0,255, NORM_MINMAX);
//        Mat matColorProcessed = new Mat();
//        cvtColor(dividedMatrix, matColorProcessed, COLOR_GRAY2BGR);
//
//        adaptiveThreshold(dividedMatrix, dividedMatrix,255,0,1,19,2);
        threshold(matGrey, matGrey,127,255, THRESH_BINARY);
        normalize(matGrey, matGrey, 0, 255, NORM_MINMAX);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        findContours(matGrey, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 0.0;
        MatOfPoint bestContour = new MatOfPoint();
        for(MatOfPoint contour : contours) {
            double area = contourArea(contour);
            if (area > 1000.0) {
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = contour;
                }
            }
        }

        List<MatOfPoint> bestContourList = new ArrayList<>();
        bestContourList.add(bestContour);

        Mat maskMatrix = zeros(new org.opencv.core.Size(matGrey.width(), matGrey.height()), CV_8U);

        drawContours(maskMatrix, bestContourList, 0, new Scalar(255.0, 255.0, 255.0, 255.0), -1);
        drawContours(maskMatrix, bestContourList, 0, new Scalar(0.0, 0.0, 0.0, 0.0), 2);

        Core.bitwise_and(matGrey, maskMatrix, matGrey);

        maskMatrix = zeros(new org.opencv.core.Size(matGrey.width() + 2, matGrey.height() + 2), CV_8U);
        floodFill(matGrey, maskMatrix, new Point(0,0), new Scalar(255, 255));

//        Canny(matGrey, matGrey, 90, 150, 3, true); // co to ten l2gradient?
//        Mat kernel = ones(3,3, CV_8U);
//        dilate(matGrey, matGrey, kernel);
//        kernel = ones(5,5, CV_8U);
//        erode(matGrey, matGrey, kernel);

        MatOfPoint corners = new MatOfPoint();
//        List<Point> cornersReduced = new ArrayList<>();

        Mat emptyMat = new Mat();
        goodFeaturesToTrack(matGrey, corners, (int) (chessboardSize.width * chessboardSize.height), 0.3, 50, emptyMat, 3, true, 0.04);

//        for(int index = 0; index < corners.toList().size() - 1; index++) {
//            if(!(sqrt(pow(corners.toArray()[index].x - corners.toArray()[index + 1].x, 2) + pow(corners.toArray()[index].y - corners.toArray()[index + 1].y, 2)) < 15)) {
//                cornersReduced.add(corners.toArray()[index]);
//            }
//        }

//        Point lowestXlowestY = new Point(9999,9999);
//        Point highestXhighestY = new Point(-9999,-9999);
//        Point highestXlowestY = new Point(-9999,9999);
//        Point lowestXhighestY = new Point(9999,-9999);
//
//        for(Point cornerPoint : corners.toArray()) {
//            if(cornerPoint.x < lowestXlowestY.x && cornerPoint.y < lowestXlowestY.y) {
//                lowestXlowestY = cornerPoint;
//            }
//            if(cornerPoint.x > highestXhighestY.x && cornerPoint.y > highestXhighestY.y) {
//                highestXhighestY = cornerPoint;
//            }
//            if(cornerPoint.x > highestXlowestY.x && cornerPoint.y < highestXlowestY.y) {
//                highestXlowestY = cornerPoint;
//            }
//            if(cornerPoint.x < lowestXhighestY.x && cornerPoint.y > lowestXhighestY.y) {
//                lowestXhighestY = cornerPoint;
//            }
//        }
//
//        drawMarker(matColor, lowestXlowestY, COLOR_BLUE, 2, 8, 5, 1);
//        drawMarker(matColor, highestXhighestY, COLOR_BLUE, 2, 8, 5, 1);
//        drawMarker(matColor, highestXlowestY, COLOR_BLUE, 2, 8, 5, 1);
//        drawMarker(matColor, lowestXlowestY, COLOR_BLUE, 2, 8, 5, 1);
//
//        putText(matColor, "" + lowestXlowestY.x + " " + lowestXlowestY.y, lowestXlowestY, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "" + lowestXlowestY.x + " " + lowestXlowestY.y, lowestXhighestY, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "lowestY" + lowestY.y, lowestY, FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//        putText(matColor, "highestY" + highestY.y, highestY, FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);

        MatOfPoint2f bestContour2f = new MatOfPoint2f(bestContour.toArray());
        RotatedRect rotatedRectangle = minAreaRect(bestContour2f);

        List<Point> orderedPoints = corners.toList();

        double angle = rotatedRectangle.angle;

        if (rotatedRectangle.size.width < rotatedRectangle.size.height) {
            angle = angle + 90;
        }

        putText(matColor, String.valueOf(angle), new Point(200,200), FONT_HERSHEY_SIMPLEX, 2, COLOR_RED);

        double finalAngle = angle;

        Collections.sort(orderedPoints, new Comparator<Point>() {
            public int compare(Point x1, Point x2) {
                double x1Prime = x1.x * cos(Math.toRadians(finalAngle)) - x1.y * sin(Math.toRadians(finalAngle));
                double y1Prime = x1.x * sin(Math.toRadians(finalAngle)) + x1.y * cos(Math.toRadians(finalAngle));
                double x2Prime = x2.x * cos(Math.toRadians(finalAngle)) - x2.y * sin(Math.toRadians(finalAngle));
                double y2Prime = x2.x * sin(Math.toRadians(finalAngle)) + x2.y * cos(Math.toRadians(finalAngle));
                return Double.compare(100 * y1Prime + 10 * x1Prime, 100 * y2Prime + 10 * x2Prime);
            }
        });

//        Point[][] cornersArray = new Point[(int) chessboardSize.height][(int) chessboardSize.width];

//        for(int i = 0, j = 0; i * j < chessboardSize.height * chessboardSize.width - 1;) {
//            cornersArray[i][j] = orderedPoints.get(i * j);
//            if(j == chessboardSize.width - 1) {
//                j = 0;
//                i++;
//            }
//        }

        for(Point corner : orderedPoints) {
            drawMarker(matColor, corner, COLOR_RED, 1, 2, 2, 1);
            putText(matColor, String.valueOf(orderedPoints.indexOf(corner)), corner, FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
        }

        return matColor;
    }

    private Mat detectCcTags(Bitmap bitmap) {

        NUMBER_OF_CCTAG_IMAGE_POINTS = 0;

        Mat matReference = new Mat();

        Mat matColor = new Mat();
        Mat matGrey = new Mat();
        Mat matGreyAdapted = new Mat();
        Mat edges = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Utils.bitmapToMat(bitmap, matColor);

        cvtColor(matColor, matGrey, COLOR_BGR2GRAY);
        //int s = matGrey.width() / 8;
        //adaptiveThreshold(matGrey, matGreyAdapted, 255, ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, s, 7.0);
        org.opencv.imgproc.Imgproc.threshold(matGrey, matGreyAdapted, 125, 255, THRESH_BINARY_INV);
        findContours(matGreyAdapted, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        //Log.d("Debug:", String.valueOf(contours.size()));

        // Draw found contours on input image
        //drawContours(matColor, contours, -1, COLOR_RED, 5);

        //Utils.matToBitmap(matColor, processedImage)
        //mImageView2.setImageBitmap(processedImage)

        int depth;
        Double contourId;

        RotatedRect elli = new RotatedRect();

        MatOfPoint2f contour2f = new MatOfPoint2f();
        org.opencv.core.Point meanCenter;
        int orderId = 0;

        List<Point> outerRingOfPoints = new ArrayList<>();
        List<Point> imagePoints = new ArrayList<>();
        List<Point> imagePointsReduced = new ArrayList<>();

        List<Double> contourIds = new ArrayList<>();

        for (int i = 0; i < contours.size(); i++) {

            List<Double> h = new ArrayList<>();

            for(double d : hierarchy.get(0, i)) {
                h.add(d);
            }

            depth = 0;

            if (h.get(2) != -1 && h.get(3) == -1) {

                depth++;
                contourId = h.get(2);
                contours.get(i).convertTo(contour2f, CV_32F);

                if(contour2f.toList().size() > 4) {

                    elli = fitEllipse(contour2f);
                    meanCenter = elli.center;

                    while(hierarchy.get(0, contourId.intValue())[2] != -1) {

                        contourIds.add(contourId);

                        depth++;
                        contourId = hierarchy.get(0, contourId.intValue())[2];
                        contours.get(contourId.intValue()).convertTo(contour2f, CV_32F);

                        if(hierarchy.get(0, contourId.intValue())[2] == -1) {
                            contourIds.add(contourId);
                        }

                        if(contour2f.toList().size() > 4) {

                            elli = fitEllipse(contour2f);

                            Point[] contour2fArray = contour2f.toArray();
                            //Log.d("Debug", String.valueOf(elli.angle));
                            if(outerRingOfPoints.isEmpty()) {
                                for (int j = 0; j < 8; j++) {
                                    Point ringPoint = contour2fArray[j * (contour2f.toList().size() / 8)];
                                    //PointF ringPointF = new PointF((float) ringPoint.x, (float) ringPoint.y);
                                    //circle(matColor, ringPoint, 3, COLOR_RED, -1);
                                    outerRingOfPoints.add(ringPoint);
                                }
                            }

                            meanCenter.x = (meanCenter.x + elli.center.x) / 2;
                            meanCenter.y = (meanCenter.y + elli.center.y) / 2;
                        }
                    }
                    //matReference = matColor;

                    if(depth == 5) {

                        orderId++;
                        contours.get(i).convertTo(contour2f, CV_32F);
                        //Log.d("Debug:", String.valueOf(contour2f.size()));
                        elli = fitEllipse(contour2f);

                        meanCenter.x = (meanCenter.x + elli.center.x) / 2;
                        meanCenter.y = (meanCenter.y + elli.center.y) / 2;

                        MatOfPoint2f contourInner2f = new MatOfPoint2f();

                        for (int j = 0; j < 8; j++) {
                            Point ringPoint = contour2f.toArray()[j * (contour2f.toList().size() / 8)];
                            double dy = ringPoint.y - meanCenter.y;
                            double dx = ringPoint.x - meanCenter.x;
                            double m = dy / dx;
                            int additionalPointsAdded = 0;
                            for(Point point : contour2f.toArray()) {
                                if(abs(dx) < 100 && (dy / abs(dy) == ((point.y - meanCenter.y) / abs(point.y - meanCenter.y))) && abs(point.x - meanCenter.x) < 1) {
                                    circle(matColor, point, 3, COLOR_RED, -1);
                                    imagePoints.add(point);
                                }
                                if(dx > 100 && abs(m * (point.x - meanCenter.x) - (point.y - meanCenter.y)) < 1) {
                                    circle(matColor, point, 3, COLOR_RED, -1);
                                    imagePoints.add(point);
                                }
                            }
                            for(Double id : contourIds) {
                                contours.get(id.intValue()).convertTo(contourInner2f, CV_32F);
                                for(Point point : contourInner2f.toArray()) {
                                    if(abs(dx) < 100 && (dy / abs(dy) == ((point.y - meanCenter.y) / abs(point.y - meanCenter.y))) && abs(point.x - meanCenter.x) < 1) {
                                        circle(matColor, point, 3, COLOR_RED, -1);
                                        imagePoints.add(point);
                                    }
                                    if(dx > 100 && abs(m * (point.x - meanCenter.x) - (point.y - meanCenter.y)) < 1) {
                                        circle(matColor, point, 3, COLOR_RED, -1);
                                        imagePoints.add(point);
                                    } /*else if((point.x - meanCenter.x) - abs(dx) < 10) {
                                        imagePoints.add(point);
                                    }*/
                                }
                            }

                        }
                        //ellipse(matColor, elli, COLOR_RED, 4);

                        circle(matColor, meanCenter, 3, COLOR_RED, -1);
                        putText(matColor, String.valueOf(orderId), new org.opencv.core.Point((int) meanCenter.x + 1, (int) meanCenter.y + 1),
                                FONT_HERSHEY_SIMPLEX, 2, COLOR_BLUE, 3);

                        for(int index = 0; index < imagePoints.size() - 1; index++) {
                            if(!(sqrt(pow(imagePoints.get(index).x - imagePoints.get(index + 1).x, 2) + pow(imagePoints.get(index).y - imagePoints.get(index + 1).y, 2)) < 15)) {
                                imagePointsReduced.add(imagePoints.get(index));
                            }
                        }
                        putText(matColor, String.valueOf(imagePointsReduced.size()), new org.opencv.core.Point( 100, 100),
                                FONT_HERSHEY_SIMPLEX, 2, COLOR_BLUE, 3);
                        NUMBER_OF_CCTAG_IMAGE_POINTS = imagePointsReduced.size();
                        Log.d("Debug:", String.valueOf(NUMBER_OF_CCTAG_IMAGE_POINTS));
                        if(NUMBER_OF_CCTAG_IMAGE_POINTS == 47 && TAKE_PHOTO == 1) {
                            File file = new File(getFilesDir(), "Ref" + System.currentTimeMillis() + ".jpg");
                            TAKE_PHOTO = 0;
                            try {
                                FileOutputStream out = new FileOutputStream(file);
                                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                                out.flush();
                                out.close();
                            } catch (Exception e) {
                            }
                        }
                    }
                }
            }
        }
        matReference = matColor;

        return matReference;
    }

    private void showAcceptedRejectedButton(boolean acceptedRejected) {
        if (acceptedRejected) {
            CameraX.unbind(preview, imageAnalysis);
            llBottom.setVisibility(View.VISIBLE);
            btnCapture.hide();
            textureView.setVisibility(View.GONE);
        } else {
            btnCapture.show();
            llBottom.setVisibility(View.GONE);
            textureView.setVisibility(View.VISIBLE);
            textureView.post(new Runnable() {
                @Override
                public void run() {
                    startCamera();
                }
            });
        }
    }


    private void updateTransform() {
        Matrix mx = new Matrix();
        float w = textureView.getMeasuredWidth();
        float h = textureView.getMeasuredHeight();

        float cX = w / 2f;
        float cY = h / 2f;

        int rotationDgr;
        int rotation = (int) textureView.getRotation();

        switch (rotation) {
            case Surface.ROTATION_0:
                rotationDgr = 0;
                break;
            case Surface.ROTATION_90:
                rotationDgr = 90;
                break;
            case Surface.ROTATION_180:
                rotationDgr = 180;
                break;
            case Surface.ROTATION_270:
                rotationDgr = 270;
                break;
            default:
                return;
        }

//        mx.postRotate((float) rotationDgr, cX, cY);
//        textureView.setTransform(mx);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()) {
            case R.id.clear:
                clearDirectory();
                return true;

            case R.id.calibrate:
                calibration();
                return true;

            case R.id.chessboard:
                currentImageProcessing = "CHESSBOARD";
                startCamera();
                return true;

            case R.id.ccTag:
                currentImageProcessing = "CCTAG";
                startCamera();
                return true;

            case R.id.assymetricCircleGrid:
                currentImageProcessing = "ASSYMETRIC_CIRCLES";
                startCamera();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {

            case R.id.btnReject:
                showAcceptedRejectedButton(false);
                break;

            case R.id.btnAccept:
                File file = new File(
                        getFilesDir(), "" + System.currentTimeMillis() + "_JDCameraX.jpg");

                imageCapture.takePicture(file, new ImageCapture.OnImageSavedListener() {
                    @Override
                    public void onImageSaved(@NonNull File file) {
                        showAcceptedRejectedButton(false);
                        try {
                            ExifInterface exif = new ExifInterface(file.getPath());
                            int orientation = exif.getAttributeInt(
                                    ExifInterface.TAG_ORIENTATION,
                                    ExifInterface.ORIENTATION_NORMAL);

                            int angle = 0;

                            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) {
                                angle = 90;
                            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_180) {
                                angle = 180;
                            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) {
                                angle = 270;
                            }

                            Matrix mat = new Matrix();
                            mat.postRotate(angle);
                            BitmapFactory.Options options = new BitmapFactory.Options();
                            options.inSampleSize = 2;

                            Bitmap bmp = BitmapFactory.decodeStream(new FileInputStream(file),
                                    null, options);
                            Bitmap bitmap = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
                                    bmp.getHeight(), mat, true);
                            FileOutputStream fout = new FileOutputStream(file);
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100,
                                    fout);

                        } catch (IOException e) {
                            Log.w("TAG", "-- Error in setting image");
                        } catch (OutOfMemoryError oom) {
                            Log.w("TAG", "-- OOM Error in setting image");
                        }
                        try {
                            Exif exif = Exif.createFromFile(file);
                            Log.d("Debug:", String.valueOf(exif.getRotation()));

                        } catch(IOException e) {

                        }
                        Toast.makeText(getApplicationContext(), "Image saved successfully in Pictures Folder", Toast.LENGTH_LONG).show();
                    }

                    @Override
                    public void onError(@NonNull ImageCapture.UseCaseError useCaseError, @NonNull String message, @Nullable Throwable cause) {

                    }
                });
                break;
        }
    }
}
