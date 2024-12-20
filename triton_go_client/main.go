package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"net/http"
	"sync"
	"time"
	triton "triton-go-client/core/grpc-client"

	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	tritonServerURL = "localhost:8001"
	modelName       = "yolov8"
)

var (
	client         triton.GRPCInferenceServiceClient
	mutex          sync.Mutex
	runningThreads = make(map[string]bool)
)

func initClient() {
	conn, err := grpc.DialContext(context.Background(), tritonServerURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Triton server: %v", err)
	}
	client = triton.NewGRPCInferenceServiceClient(conn)
}

func runInference(frame []byte, objectClass string) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	input := &triton.ModelInferRequest_InferInputTensor{
		Name:     "input",
		Datatype: "UINT8",
		Shape:    []int64{1, int64(len(frame))},
	}
	output := &triton.ModelInferRequest_InferRequestedOutputTensor{
		Name: "output",
	}

	request := &triton.ModelInferRequest{
		ModelName: modelName,
		Inputs:    []*triton.ModelInferRequest_InferInputTensor{input},
		Outputs:   []*triton.ModelInferRequest_InferRequestedOutputTensor{output},
		RawInputContents: [][]byte{
			frame,
		},
	}

	response, err := client.ModelInfer(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	detections, err := parseResponse(response, objectClass)
	return detections, err
}

func detectFromVideo(ctx *gin.Context) {
	objectClass := ctx.PostForm("object_class")
	file, err := ctx.FormFile("video_file")
	if err != nil {
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Invalid file upload"})
		return
	}

	openedFile, err := file.Open()
	if err != nil {
		ctx.JSON(http.StatusInternalServerError, gin.H{"error": "Cannot process video file"})
		return
	}
	defer openedFile.Close()

	// Process video frame-by-frame (placeholder for actual video processing)
	detections := []string{}
	maxFrames := 10 // Or implement actual frame counting logic
	for i := 0; i < maxFrames; i++ {
		frame := []byte{} // Placeholder: extract frame data
		frameDetections, err := runInference(frame, objectClass)
		if err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		detections = append(detections, frameDetections...)
	}

	ctx.JSON(http.StatusOK, gin.H{"detections": detections})
}

func detectFromLiveVideo(ctx *gin.Context) {
	cameraIP := ctx.PostForm("camera_ip")
	objectClass := ctx.PostForm("object_class")

	mutex.Lock()
	if runningThreads[cameraIP] {
		mutex.Unlock()
		ctx.JSON(http.StatusBadRequest, gin.H{"error": "Detection already running for this camera"})
		return
	}
	runningThreads[cameraIP] = true
	mutex.Unlock()

	go func() {
		defer func() {
			mutex.Lock()
			delete(runningThreads, cameraIP)
			mutex.Unlock()
		}()

		// Placeholder for actual camera stream processing
		for {
			frame := []byte{} // Placeholder: fetch frame from camera
			_, err := runInference(frame, objectClass)
			if err != nil {
				log.Printf("Error running inference for camera %s: %v", cameraIP, err)
				break
			}
		}
	}()

	ctx.JSON(http.StatusOK, gin.H{"message": "Detection started for live video"})
}

func float32FromBytes(data []byte) float32 {
	return math.Float32frombits(binary.LittleEndian.Uint32(data))
}

func classLabelToIndex(label string) int {
	labelMap := map[string]int{
		"person":         0,
		"bicycle":        1,
		"car":            2,
		"motorcycle":     3,
		"airplane":       4,
		"bus":            5,
		"train":          6,
		"truck":          7,
		"boat":           8,
		"traffic light":  9,
		"fire hydrant":   10,
		"stop sign":      11,
		"parking meter":  12,
		"bench":          13,
		"bird":           14,
		"cat":            15,
		"dog":            16,
		"horse":          17,
		"sheep":          18,
		"cow":            19,
		"elephant":       20,
		"bear":           21,
		"zebra":          22,
		"giraffe":        23,
		"backpack":       24,
		"umbrella":       25,
		"handbag":        26,
		"tie":            27,
		"suitcase":       28,
		"frisbee":        29,
		"skis":           30,
		"snowboard":      31,
		"sports ball":    32,
		"kite":           33,
		"baseball bat":   34,
		"baseball glove": 35,
		"skateboard":     36,
		"surfboard":      37,
		"tennis racket":  38,
		"bottle":         39,
		"wine glass":     40,
		"cup":            41,
		"fork":           42,
		"knife":          43,
		"spoon":          44,
		"bowl":           45,
		"banana":         46,
		"apple":          47,
		"sandwich":       48,
		"orange":         49,
		"broccoli":       50,
		"carrot":         51,
		"hot dog":        52,
		"pizza":          53,
		"donut":          54,
		"cake":           55,
		"chair":          56,
		"couch":          57,
		"potted plant":   58,
		"bed":            59,
		"dining table":   60,
		"toilet":         61,
		"tv":             62,
		"laptop":         63,
		"mouse":          64,
		"remote":         65,
		"keyboard":       66,
		"cell phone":     67,
		"microwave":      68,
		"oven":           69,
		"toaster":        70,
		"sink":           71,
		"refrigerator":   72,
		"book":           73,
		"clock":          74,
		"vase":           75,
		"scissors":       76,
		"teddy bear":     77,
		"hair drier":     78,
		"toothbrush":     79,
	}

	return labelMap[label]
}

func parseResponse(response *triton.ModelInferResponse, objectClass string) ([]string, error) {
	// Get the raw output tensor
	rawOutput := response.RawOutputContents[0]

	const detectionStride = 85 // YOLOv8 typically uses 85 values per detection: 4 box, 1 confidence, 80 classes
	detectionThreshold := float32(0.5)
	objectClassIndex := classLabelToIndex(objectClass)
	detections := []string{}

	numDetections := len(rawOutput) / (detectionStride * 4) // Assuming float32 values, each detection takes 85 * 4 bytes
	for i := 0; i < numDetections; i++ {
		offset := i * detectionStride * 4
		xCenter := float32FromBytes(rawOutput[offset : offset+4])
		yCenter := float32FromBytes(rawOutput[offset+4 : offset+8])
		width := float32FromBytes(rawOutput[offset+8 : offset+12])
		height := float32FromBytes(rawOutput[offset+12 : offset+16])
		confidence := float32FromBytes(rawOutput[offset+16 : offset+20])

		if confidence < detectionThreshold {
			continue
		}

		// Get class probabilities and the highest-scoring class
		maxProb := float32(0)
		detectedClass := -1
		for j := 0; j < 80; j++ { // Assuming 80 classes
			classProb := float32FromBytes(rawOutput[offset+20+(j*4) : offset+24+(j*4)])
			if classProb > maxProb {
				maxProb = classProb
				detectedClass = j
			}
		}

		// Check if the detected class matches the requested object class
		if detectedClass == objectClassIndex {
			xMin := xCenter - width/2
			yMin := yCenter - height/2
			xMax := xCenter + width/2
			yMax := yCenter + height/2

			detections = append(detections, fmt.Sprintf(
				"BoundingBox: [%.2f, %.2f, %.2f, %.2f], Confidence: %.2f",
				xMin, yMin, xMax, yMax, confidence,
			))
		}
	}

	return detections, nil
}

func main() {
	initClient()

	router := gin.Default()
	router.POST("/detect_objects_in_video", detectFromVideo)
	router.POST("/detect_objects_in_live_video", detectFromLiveVideo)

	router.Run(":8080")
}
