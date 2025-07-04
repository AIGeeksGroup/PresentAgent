import { createRouter, createWebHistory } from 'vue-router'
import UploadComponent from '../components/Upload.vue'
import GenerateComponent from '../components/Generate.vue'
import PptToVideo from '../components/PptToVideo.vue'

// ... existing routes ...

const routes = [
  {
    path: '/',
    name: 'Upload',
    component: UploadComponent
  },
  // Removed Doc route
  {
    path: '/generate',
    name: 'Generate',
    component: GenerateComponent
  },
  {
    path: '/ppt-to-video',
    name: 'PptToVideo',
    component: PptToVideo
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
