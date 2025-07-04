<template>
  <div class="generate-container">
    <p class="task-id">Task ID: {{ taskId }}</p>
    <progress :value="progress" max="100" class="progress-bar"></progress>
    <p class="status-message">{{ statusMessage }}</p>
    <a v-if="downloadLink" :href="downloadLink" :download="filename" class="download-link">Download PPTX</a>
    <input v-model="feedback" placeholder="Enter your feedback with contact information" class="feedback-input" />
    <button @click="submitFeedback" class="feedback-button">Submit Feedback</button>
    <button class="back-button" @click="goBack">Back</button>
  </div>
</template>

<script>
export default {
  name: 'GenerateComponent',
  data() {
    return {
      progress: 0,
      statusMessage: 'Starting...',
      downloadLink: '',
      taskId: history.state.taskId,
      feedback: '',
      filename: 'final.pptx',
      socket: null,
    }
  },
  created() {
    this.startGeneration()
  },
  beforeUnmount() {
    this.closeSocket()
  },
  methods: {
    goBack() {
      this.$router.push('/')
    },
    async startGeneration() {
      console.log("Connecting to websocket", `/wsapi/${this.taskId}`)
      this.socket = new WebSocket(`/wsapi/${this.taskId}`)

      this.socket.onmessage = (event) => {
        console.log("Socket Received message:", event.data)
        const data = JSON.parse(event.data)
        this.progress = data.progress
        this.statusMessage = data.status
        if (data.progress >= 100) {
          this.closeSocket()
          this.fetchDownloadLink()
        }
      }
      this.socket.onerror = (error) => {
        console.error("WebSocket error:", error)
        this.statusMessage = 'WebSocket connection failed.'
        this.closeSocket()
      }
    },
    async fetchDownloadLink() {
      try {
        const downloadResponse = await this.$axios.get('/api/download', { params: { task_id: this.taskId }, responseType: 'blob' })
        this.downloadLink = URL.createObjectURL(downloadResponse.data)
        this.filename = "ppagent_" + this.taskId.replace('/', '_') + '.pptx'
      } catch (error) {
        console.error("Download error:", error)
        this.statusMessage += '\nFailed to continue the task.'
      }
    },
    async submitFeedback() {
      if (!this.feedback) {
        alert('Please enter your feedback with contact information.')
        return
      }
      try {
        await this.$axios.post('/api/feedback', { feedback: this.feedback, task_id: this.taskId })
        this.statusMessage = 'Feedback submitted successfully.'
        this.feedback = ''
      } catch (error) {
        console.error("Feedback submission error:", error)
        this.statusMessage = 'Failed to submit feedback.'
      }
    },
    closeSocket() {
      if (this.socket) {
        this.socket.close()
        this.socket = null
      }
    }
  }
}
</script>

<style scoped>
.generate-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
}

.progress-bar {
  width: 100%;
  height: 20px;
  margin-bottom: 10px;
  appearance: none;
  background-color: #f3f3f3;
}

.task-id {
  font-size: 1.2em;
  margin-bottom: 10px;
  color: #df8638;
}

.progress-bar::-webkit-progress-bar {
  background-color: #f3f3f3;
}

.progress-bar::-webkit-progress-value {
  background-color: #42b983;
}

.status-message {
  font-size: 1.2em;
  margin-bottom: 10px;
  color: #333;
}

.download-link {
  font-size: 1em;
  color: #42b983;
  text-decoration: none;
  border: 1px solid #42b983;
  padding: 8px 16px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.download-link:hover {
  background-color: #42b983;
  color: #fff;
}

.feedback-input {
  margin-top: 10px;
  padding: 8px;
  width: 100%;
  box-sizing: border-box;
}

.feedback-button {
  margin-top: 10px;
  padding: 8px 16px;
  background-color: #42b983;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.feedback-button:hover {
  background-color: #369b72;
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
</style>
