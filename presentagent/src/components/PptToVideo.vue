<template>
  <div class="ppt-to-video-container">
    <div class="upload-options">
      <!-- Title -->
      <div class="page-title">
        <h2>PPT to Presentation</h2>
        <p>Upload your PPT file and we will generate a presentation video with voice-over narration</p>
        <button class="back-button" @click="goBack">Back</button>
      </div>

      <!-- File Upload Area -->
      <div class="upload-section">
        <label for="ppt-upload" class="upload-label">
          <div class="upload-icon">📄</div>
          <div class="upload-text">
            <span v-if="!pptFile">Click to upload PPT file</span>
            <span v-else>{{ pptFile.name }} ✔️</span>
          </div>
        </label>
        <input type="file" id="ppt-upload" @change="handleFileUpload" accept=".pptx,.ppt" />
      </div>

      <!-- Action Button -->
      <button @click="startConversion" :disabled="!pptFile || isProcessing" class="convert-button">
        <span v-if="!isProcessing">Start Conversion</span>
        <span v-else>Processing...</span>
      </button>

      <!-- Progress Bar Section -->
      <div v-if="isProcessing" class="progress-section">
        <div class="progress-info">
          <h3>Processing your PPT...</h3>
          <p>Current progress: {{ currentSlide }}/{{ totalSlides }} pages</p>
        </div>

        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: progressPercentage + '%' }"></div>
          </div>
          <span class="progress-text">{{ progressPercentage.toFixed(2) }}%</span> <!-- 保留两位小数 -->
        </div>

        <div class="processing-steps">
          <div class="step" :class="{ active: currentStep >= 1, completed: currentStep > 1 }">
            <span class="step-number">1</span>
            <span class="step-text">Parsing PPT file</span>
          </div>
          <div class="step" :class="{ active: currentStep >= 2, completed: currentStep > 2 }">
            <span class="step-number">2</span>
            <span class="step-text">Generating voice</span>
          </div>
          <div class="step" :class="{ active: currentStep >= 3, completed: currentStep > 3 }">
            <span class="step-number">3</span>
            <span class="step-text">Composing video</span>
          </div>
        </div>
      </div>

      <!-- Result Section -->
      <div v-if="videoUrl && progressPercentage === 100" class="result-section">
        <h3>Conversion completed!</h3>
        <div class="video-preview">
          <video controls :src="videoUrl" width="100%" height="300">
            Your browser does not support video playback.
          </video>
        </div>
        <div class="download-section">
          <a :href="videoUrl" download class="download-button">Download Video</a>
          <button @click="resetForm" class="reset-button">Convert Again</button>
        </div>
      </div>

      <!-- Error Message -->
      <div v-if="errorMessage" class="error-message">
        <p>{{ errorMessage }}</p>
        <button @click="resetForm" class="reset-button">Retry</button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'PptToVideo',
  data() {
    return {
      pptFile: null,
      isProcessing: false,
      currentSlide: 0,
      totalSlides: 0,
      currentStep: 0,
      videoUrl: null,
      errorMessage: null,
      progressPercentage: 0
    }
  },
  methods: {
    goBack() {
      this.$router.push('/')
    },
    handleFileUpload(event) {
      const file = event.target.files[0]
      if (file) {
        this.pptFile = file
        this.errorMessage = null
      }
    },

    async startConversion() {
      if (!this.pptFile) {
        this.errorMessage = 'Please upload a PPT file'
        return
      }

      this.isProcessing = true
      this.currentStep = 1
      this.errorMessage = null
      this.videoUrl = null

      const formData = new FormData()
      formData.append('pptFile', this.pptFile)

      try {
        // Start conversion request
        const response = await this.$axios.post('/api/ppt-to-video', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        const taskId = response.data.task_id

        this.setupWebSocket(taskId)

      } catch (error) {
        console.error("转换失败:", error)
        this.errorMessage = "转换失败，请重试"
        this.isProcessing = false
      }
    },

    setupWebSocket(taskId) {
      const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:'
      const wsUrl   = `${wsProto}//${location.host}/wsapi/ppt-to-video/${taskId}`
      this.websocket = new WebSocket(wsUrl)

      this.websocket.onopen = () => {
        console.log("WebSocket连接已建立")
      }

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data)
        this.currentSlide = data.current_slide || 0
        this.totalSlides = data.total_slides || 0
        this.currentStep = data.current_step || 1
        this.progressPercentage = data.progress_percentage || 0

        if (data.status === "completed") {
          this.websocket.close()
          this.videoUrl = data.video_url
          this.isProcessing = false
        } else if (data.status === "failed") {
          this.websocket.close()
          this.errorMessage = data.error_message || "转换失败"
          this.isProcessing = false
        }
      }

      this.websocket.onclose = () => {
        console.log("WebSocket连接已关闭")
        if (this.isProcessing) { // 如果仍在处理中，说明连接异常断开
          this.errorMessage = "与服务器的连接已断开，请重试"
          this.isProcessing = false
        }
      }

      this.websocket.onerror = (error) => {
        console.error("WebSocket错误:", error)
        this.errorMessage = "WebSocket连接发生错误，请重试"
        this.isProcessing = false
      }
    },

    resetForm() {
      if (this.websocket) {
        this.websocket.close()
      }
      this.pptFile = null
      this.isProcessing = false
      this.currentSlide = 0
      this.totalSlides = 0
      this.currentStep = 0
      this.videoUrl = null
      this.errorMessage = null
      this.progressPercentage = 0

      // 重置文件输入
      const fileInput = document.getElementById("ppt-upload")
      if (fileInput) {
        fileInput.value = ""
      }
    }
  },
  beforeUnmount() {
    if (this.websocket) {
      this.websocket.close()
    }
  }
}
</script>

<style scoped>
/* Styles unchanged (already English) */
.ppt-to-video-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  min-height: 100%;
  background-color: #f0f8ff;
  padding: 20px;
  box-sizing: border-box;
}

.upload-options {
  display: flex;
  flex-direction: column;
  gap: 30px;
  width: 100%;
  max-width: 800px;
}

.page-title {
  text-align: center;
  color: #35495e;
}

.page-title h2 {
  font-size: 28px;
  margin-bottom: 10px;
  color: #35495e;
}

.page-title p {
  font-size: 16px;
  color: #666;
  margin: 0;
}

.upload-section {
  display: flex;
  justify-content: center;
}

.upload-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background-color: #42b983;
  color: white;
  padding: 40px;
  border-radius: 10px;
  cursor: pointer;
  transition: background-color 0.3s, transform 0.2s;
  min-width: 300px;
  min-height: 150px;
}

.upload-label:hover {
  background-color: #369870;
  transform: scale(1.02);
}

.upload-icon {
  font-size: 48px;
  margin-bottom: 15px;
}

.upload-text {
  font-size: 18px;
  text-align: center;
}

.upload-section input[type="file"] {
  display: none;
}

.convert-button {
  background-color: #35495e;
  color: white;
  padding: 15px 40px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-size: 18px;
  font-weight: 700;
  transition: background-color 0.3s, transform 0.2s;
  align-self: center;
}

.convert-button:hover:not(:disabled) {
  background-color: #2c3e50;
  transform: scale(1.05);
}

.convert-button:disabled {
  background-color: #bdc3c7;
  cursor: not-allowed;
  transform: none;
}

.progress-section {
  background-color: white;
  padding: 30px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.progress-info {
  text-align: center;
  margin-bottom: 20px;
}

.progress-info h3 {
  color: #35495e;
  margin-bottom: 10px;
}

.progress-info p {
  color: #666;
  margin: 0;
}

.progress-bar-container {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 30px;
}

.progress-bar {
  flex: 1;
  height: 20px;
  background-color: #ecf0f1;
  border-radius: 10px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #42b983, #369870);
  transition: width 0.3s ease;
}

.progress-text {
  font-weight: bold;
  color: #35495e;
  min-width: 50px;
}

.processing-steps {
  display: flex;
  justify-content: space-between;
  gap: 20px;
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  padding: 20px;
  border-radius: 8px;
  background-color: #f8f9fa;
  transition: all 0.3s ease;
}

.step.active {
  background-color: #e3f2fd;
  border: 2px solid #42b983;
}

.step.completed {
  background-color: #e8f5e8;
  border: 2px solid #4caf50;
}

.step-number {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background-color: #bdc3c7;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  margin-bottom: 10px;
}

.step.active .step-number {
  background-color: #42b983;
}

.step.completed .step-number {
  background-color: #4caf50;
}

.step-text {
  font-size: 14px;
  text-align: center;
  color: #666;
}

.step.active .step-text,
.step.completed .step-text {
  color: #35495e;
  font-weight: bold;
}

.result-section {
  background-color: white;
  padding: 30px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.result-section h3 {
  color: #4caf50;
  margin-bottom: 20px;
}

.video-preview {
  margin-bottom: 20px;
  border-radius: 10px;
  overflow: hidden;
  position: relative;
  width: 100%; /* 让视频容器的宽度自适应 */
  height: 300px; /* 你可以调整容器的高度，根据需要调整 */
}

.video-preview video {
  width: 100%;
  height: 100%;
  object-fit: contain; /* 保持视频纵横比且不裁剪 */
}

.download-section {
  display: flex;
  gap: 15px;
  justify-content: center;
}

.download-button {
  background-color: #4caf50;
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  text-decoration: none;
  font-size: 16px;
  font-weight: 600;
  transition: background-color 0.3s;
}

.download-button:hover {
  background-color: #45a049;
}

.reset-button {
  background-color: #95a5a6;
  color: white;
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-weight: 600;
  transition: background-color 0.3s;
}

.reset-button:hover {
  background-color: #7f8c8d;
}

.back-button {
  position: absolute;
  left: 0;
  top: 0;
  transform: translateY(10px);
  background: #95a5a6;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 10px 24px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);
  transition: background-color 0.3s, transform 0.2s;
}


.back-button:hover {
  background: #7f8c8d;
  transform: translateY(10px) scale(1.05);
}

.error-message {
  background-color: #ffebee;
  border: 1px solid #f44336;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  color: #c62828;
}

.error-message p {
  margin-bottom: 15px;
}

@media (max-width: 768px) {
  .upload-options {
    max-width: 100%;
  }

  .upload-label {
    min-width: 250px;
    min-height: 120px;
    padding: 30px;
  }

  .processing-steps {
    flex-direction: column;
    gap: 15px;
  }

  .download-section {
    flex-direction: column;
  }

  .download-button,
  .reset-button {
    width: 100%;
  }
}

</style>
