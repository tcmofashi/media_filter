<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="isOpen" class="media-preview" @click.self="$emit('close')">
        <!-- Backdrop -->
        <div class="backdrop" @click="$emit('close')"></div>
        
        <!-- Content -->
        <div class="preview-content">
          <!-- Header -->
          <div class="preview-header">
            <div class="nav-section">
              <button 
                class="nav-btn prev" 
                @click="handlePrevious"
                :disabled="currentIndex === 0"
                title="上一个 (A / ←)"
              >
                ←
              </button>
              <span class="counter">{{ currentIndex + 1 }} / {{ files.length }}</span>
              <button 
                class="nav-btn next"
                @click="handleNext(true)"
                :disabled="currentIndex >= files.length - 1"
                :title="nextActionTitle"
              >
                →
              </button>
            </div>
            
            <span class="file-name">{{ currentFile?.name }}</span>
            
            <button class="close-btn" @click="$emit('close')" title="关闭 (Esc)">✕</button>
          </div>
          
          
          <!-- Main Area: Media + Score Panel -->
          <div class="main-area">
            <!-- Media Display -->
            <div class="media-container">
              <!-- Image preview -->
              <div 
                v-if="isImage" 
                class="image-wrapper"
                :style="imageTransformStyle"
                @wheel="handleZoom"
              >
                <img :src="getThumbnailUrl(currentFile?.path)" :alt="currentFile?.name" draggable="false" />
              </div>
              
              <!-- Video screenshots view (default for videos) -->
              <div 
                v-else-if="!isVideoPlaying" 
                class="video-screenshots-wrapper"
              >
                <div v-if="screenshotsLoading" class="screenshots-loading">
                  <div class="spinner"></div>
                  <span>加载截图中...</span>
                </div>
                <img 
                  v-else-if="screenshotsData?.screenshot_url" 
                  :src="screenshotsData.screenshot_url"
                  class="stitched-screenshot"
                  :alt="currentFile?.name"
                  draggable="false"
                />
                <!-- Play button overlay -->
                <button class="play-video-btn" @click="startVideoPlayback" title="播放视频">
                  <span class="play-icon">▶</span>
                  <span class="play-text">播放视频</span>
                </button>
              </div>
              
              <!-- Video player view -->
              <div v-else class="video-player-wrapper">
                <button class="back-to-screenshots-btn" @click="stopVideoPlayback" title="返回截图">
                  ← 返回截图
                </button>
                <video
                  ref="videoRef"
                  controls
                  preload="auto"
                  playsinline
                >
                  您的浏览器不支持视频播放
                </video>
              </div>
            </div>
            
            <!-- Right Side: Score Panel -->
            <div class="score-panel">
              <div class="score-title">评分</div>
              <div class="score-buttons">
                <button
                  v-for="n in 10"
                  :key="n-1"
                  class="score-btn"
                  :class="{ active: score === n-1 }"
                  @click="quickScore(n-1)"
                >
                  {{ n-1 }}
                </button>
              </div>
              <div class="score-hint">点击评分<br/>自动下一张</div>
            </div>
          </div>
          
          <!-- Controls -->
          <div class="preview-controls">
            <button 
              class="control-btn next-file"
              @click="handleNext(true)"
              :disabled="currentIndex >= files.length - 1"
            >
              下一个 →
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted, toRaw, nextTick } from 'vue'
import Hls from 'hls.js'
import { getThumbnailUrl, getStreamUrl, getHlsPlaylistUrl, getScreenshots } from '../api/index.js'

const props = defineProps({
  isOpen: {
    type: Boolean,
    default: false
  },
  files: {
    type: Array,
    default: () => []
  },
  currentIndex: {
    type: Number,
    default: 0
  },
  labels: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['close', 'navigate', 'save'])

// Image extensions check
const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']

// Video formats that need transcoding for browser compatibility
const needsTranscodingExts = ['.mkv', '.avi', '.flv', '.wmv', '.m4v', '.mov']

// Reference to video element
const videoRef = ref(null)
let hlsInstance = null // HLS.js instance

// Check if video needs transcoding (non-native formats)
function needsTranscoding(filePath) {
  if (!filePath) return false
  const ext = filePath.slice(filePath.lastIndexOf('.')).toLowerCase()
  // MP4 and WebM can be played natively in most browsers
  const nativeExts = ['.mp4', '.webm']
  return !nativeExts.includes(ext)
}

// State
const zoomLevel = ref(1)
const score = ref(5) // 0-9 scale, default 5 (mid)
let zoomRafId = null // requestAnimationFrame ID for zoom throttling

// Video playback state
const isVideoPlaying = ref(false)
const screenshotsData = ref(null)
const screenshotsLoading = ref(false)

function toPresetScore(scoreValue) {
  const numericScore = Number(scoreValue)
  if (!Number.isFinite(numericScore)) return null
  return Math.max(0, Math.min(9, Math.round(numericScore)))
}

// Load screenshots for video preview
async function loadVideoScreenshots() {
  const filePath = currentFile.value?.path
  if (!filePath) return
  
  // Skip if already cached for THIS video
  if (screenshotsData.value?.filePath === filePath && 
      screenshotsData.value?.status === 'cached') {
    return
  }
  
  screenshotsLoading.value = true
  screenshotsData.value = null
  
  try {
    const result = await getScreenshots(filePath)
    if (result.status === 'cached' && result.screenshot_url) {
      screenshotsData.value = { ...result, filePath }
      screenshotsLoading.value = false
    } else if (result.status === 'processing') {
      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const pollResult = await getScreenshots(filePath)
          if (pollResult.status === 'cached' && pollResult.screenshot_url) {
            screenshotsData.value = { ...pollResult, filePath }
            screenshotsLoading.value = false
            clearInterval(pollInterval)
          }
        } catch (e) {
          clearInterval(pollInterval)
          screenshotsLoading.value = false
        }
      }, 1000)
      // Timeout after 30 seconds
      setTimeout(() => {
        clearInterval(pollInterval)
        screenshotsLoading.value = false
      }, 30000)
    }
  } catch (e) {
    console.error('Failed to load screenshots:', e)
    screenshotsLoading.value = false
  }
}

// Video playback control
async function startVideoPlayback() {
  isVideoPlaying.value = true
  
  // Wait for video element to be mounted
  await nextTick()
  
  const video = videoRef.value
  const filePath = currentFile.value?.path
  
  if (!video || !filePath) return
  
  if (needsTranscoding(filePath)) {
    // Use HLS for non-native formats
    const hlsUrl = getHlsPlaylistUrl(filePath)
    
    if (Hls.isSupported()) {
      // Destroy previous instance if exists
      if (hlsInstance) {
        hlsInstance.destroy()
      }
      
      hlsInstance = new Hls({
        enableWorker: true,
        lowLatencyMode: false,
      })
      
      hlsInstance.loadSource(hlsUrl)
      hlsInstance.attachMedia(video)
      
      hlsInstance.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(e => console.log('Autoplay prevented:', e))
      })
      
      hlsInstance.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          console.error('HLS fatal error:', data)
          hlsInstance.destroy()
        }
      })
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari native HLS support
      video.src = hlsUrl
      video.play().catch(e => console.log('Autoplay prevented:', e))
    }
  } else {
    // Native formats - direct playback
    video.src = getStreamUrl(filePath)
    video.play().catch(e => console.log('Autoplay prevented:', e))
  }
}

function stopVideoPlayback() {
  // Destroy HLS instance
  if (hlsInstance) {
    hlsInstance.destroy()
    hlsInstance = null
  }
  
  // Pause and reset video
  if (videoRef.value) {
    videoRef.value.pause()
    videoRef.value.currentTime = 0
    videoRef.value.src = ''
  }
  
  isVideoPlaying.value = false
}

// Computed
const currentFile = computed(() => props.files[props.currentIndex] || null)
const shouldAutoLabelOnAdvance = computed(() => currentFile.value?.inference_score != null)
const nextActionTitle = computed(() => (
  shouldAutoLabelOnAdvance.value
    ? '标注当前分数并前进 (D / →)'
    : '下一个 (D / →)'
))

// Cache extension check results
const extensionCache = new Map()

const isImage = computed(() => {
  if (!currentFile.value?.name) return false
  const path = currentFile.value.path
  
  if (extensionCache.has(path)) {
    return extensionCache.get(path) === 'image'
  }
  
  const ext = currentFile.value.name.slice(currentFile.value.name.lastIndexOf('.')).toLowerCase()
  const result = imageExtensions.includes(ext)
  extensionCache.set(path, result ? 'image' : 'video')
  return result
})

const imageTransformStyle = computed(() => ({
  transform: `scale(${zoomLevel.value})`
}))

// Pre-computed label map (updated when labels change)
const labelMap = computed(() => {
  const map = new Map()
  const rawLabels = toRaw(props.labels)
  // Use for loop for better performance than forEach
  for (let i = 0; i < rawLabels.length; i++) {
    const label = rawLabels[i]
    map.set(label.media_path, label.score)
  }
  return map
})

const currentScore = computed(() => {
  if (!currentFile.value) return 5

  const saved = labelMap.value.get(currentFile.value.path)
  const labeledScore = toPresetScore(saved)
  if (labeledScore !== null) return labeledScore

  const inferenceScore = toPresetScore(currentFile.value.inference_score)
  if (inferenceScore !== null) return inferenceScore

  return 5
})

// Methods - Zoom with requestAnimationFrame throttling
function handleZoom(event) {
  event.preventDefault()
  
  // Cancel previous frame if pending
  if (zoomRafId) {
    return // Skip if already waiting for frame
  }
  
  zoomRafId = requestAnimationFrame(() => {
    zoomRafId = null
    if (event.deltaY < 0) {
      zoomLevel.value = Math.min(zoomLevel.value + 0.2, 4)
    } else {
      zoomLevel.value = Math.max(zoomLevel.value - 0.2, 0.5)
    }
  })
}

function handlePrevious() {
  if (props.currentIndex > 0) {
    emit('navigate', props.currentIndex - 1)
  }
}

function handleNext(saveCurrent = false) {
  if (!currentFile.value) return

  if (saveCurrent && shouldAutoLabelOnAdvance.value) {
    saveLabel()
  }

  if (props.currentIndex < props.files.length - 1) {
    emit('navigate', props.currentIndex + 1)
  }
}

function saveLabel() {
  emit('save', { mediaPath: currentFile.value?.path, score: score.value })
}

function quickScore(newScore) {
  score.value = newScore
  saveLabel()
  // Auto navigate to next file after a short delay
  setTimeout(() => {
    if (props.currentIndex < props.files.length - 1) {
      emit('navigate', props.currentIndex + 1)
    }
  }, 150)
}

watch(currentScore, (value) => {
  score.value = value
}, { immediate: true })

// Watchers - single watcher for index changes
watch(() => props.currentIndex, () => {
  if (props.isOpen) {
    zoomLevel.value = 1
    // Reset video playback state and destroy HLS
    if (hlsInstance) {
      hlsInstance.destroy()
      hlsInstance = null
    }
    isVideoPlaying.value = false
    // Reset screenshots data for new video
    screenshotsData.value = null
    // Load screenshots for videos
    if (!isImage.value && currentFile.value?.path) {
      loadVideoScreenshots()
    }
  }
})

// Also watch for isOpen changes to load initial screenshots
watch(() => props.isOpen, (isOpen) => {
  if (isOpen && !isImage.value && currentFile.value?.path) {
    loadVideoScreenshots()
  }
})

// Keyboard handler
function handleKeydown(event) {
  if (!props.isOpen) return
  
  switch (event.key) {
    case 'Escape':
      emit('close')
      event.preventDefault()
      break
    case 'a':
    case 'A':
    case 'ArrowLeft':
      if (props.currentIndex > 0) {
        emit('navigate', props.currentIndex - 1)
      }
      event.preventDefault()
      break
    case 'd':
    case 'D':
    case 'ArrowRight':
      handleNext(true)
      event.preventDefault()
      break
  }
}

// Lifecycle
onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  // Clean up HLS instance
  if (hlsInstance) {
    hlsInstance.destroy()
    hlsInstance = null
  }
})
</script>

<style scoped>
/* Use system fonts for better performance */
.media-preview {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.95);
}

/* Preview Content Container */
.preview-content {
  position: relative;
  width: 100%;
  max-width: 1200px;
  height: 90vh;
  margin: 20px;
  background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 16px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-shadow: 
    0 32px 64px rgba(0, 0, 0, 0.6),
    0 0 0 1px rgba(66, 184, 131, 0.1),
    0 0 80px rgba(66, 184, 131, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Header */
.preview-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  background: rgba(0, 0, 0, 0.5);
  border-bottom: 1px solid rgba(66, 184, 131, 0.15);
}

.nav-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.nav-btn {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(66, 184, 131, 0.1);
  border: 1px solid rgba(66, 184, 131, 0.3);
  border-radius: 10px;
  color: #42b883;
  font-size: 18px;
  font-family: 'JetBrains Mono', monospace;
  cursor: pointer;
  transition: all 0.08s ease-out;
}

.nav-btn:hover:not(:disabled) {
  background: #42b883;
  color: #0a0a0a;
}

.nav-btn:disabled {
  opacity: 0.25;
  cursor: not-allowed;
  border-color: rgba(255, 255, 255, 0.1);
  color: #666;
}

.counter {
  font-family: 'Orbitron', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: #8b92a8;
  min-width: 70px;
  text-align: center;
  letter-spacing: 1px;
}

.file-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 500;
  color: #e4e6f0;
  max-width: 400px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.close-btn {
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(220, 53, 69, 0.1);
  border: 1px solid rgba(220, 53, 69, 0.3);
  border-radius: 10px;
  color: #ff6b6b;
  font-size: 18px;
  font-family: 'Orbitron', sans-serif;
  cursor: pointer;
  transition: all 0.08s ease-out;
}

.close-btn:hover {
  background: #ff6b6b;
  color: #0a0a0a;
}

/* Media Container */
.media-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: #0d0d12;
}

/* Main Area - Horizontal container for media + score panel */
.main-area {
  flex: 1;
  display: flex;
  overflow: hidden;
}

/* Score Panel - Right Side */
.score-panel {
  width: 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px 12px;
  background: rgba(0, 0, 0, 0.6);
  border-left: 1px solid rgba(66, 184, 131, 0.15);
  gap: 12px;
}

.score-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
  color: #8b92a8;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.score-buttons {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
}

.score-btn {
  width: 100%;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  color: #ccc;
  font-family: 'Orbitron', sans-serif;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.1s ease;
}

.score-btn:hover {
  background: rgba(66, 184, 131, 0.2);
  border-color: rgba(66, 184, 131, 0.5);
  color: #42b883;
}

.score-btn.active {
  background: #42b883;
  border-color: #42b883;
  color: #0a0a0a;
}

.score-hint {
  font-size: 10px;
  color: #666;
  text-align: center;
  line-height: 1.4;
  margin-top: auto;
}
.media-container {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: #0d0d12;
}

.image-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  max-width: 100%;
  max-height: 100%;
  }

.image-wrapper img {
  max-width: 90%;
  max-height: 70vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  user-select: none;
  -webkit-user-drag: none;
  transition: transform 0.05s ease;
}

.media-container video {
  max-width: 90%;
  max-height: 70vh;
  border-radius: 8px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

/* Video Screenshots View */
.video-screenshots-wrapper {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.screenshots-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  color: #42b883;
  font-size: 14px;
}

.screenshots-loading .spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(66, 184, 131, 0.3);
  border-top-color: #42b883;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.stitched-screenshot {
  max-width: 90%;
  max-height: 70vh;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
  user-select: none;
  -webkit-user-drag: none;
}

/* Play Video Button */
.play-video-btn {
  position: absolute;
  bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: linear-gradient(135deg, rgba(66, 184, 131, 0.9) 0%, rgba(54, 158, 110, 0.9) 100%);
  border: none;
  border-radius: 8px;
  color: #0a0a0a;
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease-out;
  box-shadow: 0 4px 20px rgba(66, 184, 131, 0.4);
}

.play-video-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 30px rgba(66, 184, 131, 0.5);
}

.play-icon {
  font-size: 18px;
}

/* Video Player Wrapper */
.video-player-wrapper {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
}

.back-to-screenshots-btn {
  position: absolute;
  top: 10px;
  left: 10px;
  padding: 8px 16px;
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(66, 184, 131, 0.3);
  border-radius: 6px;
  color: #42b883;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.1s ease;
  z-index: 10;
}

.back-to-screenshots-btn:hover {
  background: rgba(66, 184, 131, 0.2);
}

/* Preview Controls */
.preview-controls {
  display: flex;
  justify-content: center;
  padding: 16px 24px;
  background: rgba(0, 0, 0, 0.2);
  border-top: 1px solid rgba(255, 255, 255, 0.03);
}

.control-btn {
  padding: 12px 28px;
  background: linear-gradient(135deg, rgba(66, 184, 131, 0.15) 0%, rgba(66, 184, 131, 0.05) 100%);
  border: 1px solid rgba(66, 184, 131, 0.3);
  border-radius: 8px;
  color: #42b883;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.5px;
  cursor: pointer;
  transition: all 0.08s ease-out;
}

.control-btn:hover:not(:disabled) {
  background: #42b883;
  color: #0a0a0a;
}

.control-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}


/* Transitions */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.1s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

/* Responsive */
@media (max-width: 768px) {
  .preview-content {
    height: 100vh;
    max-width: 100%;
    margin: 0;
    border-radius: 0;
  }
  
  .preview-header {
    padding: 16px;
    flex-wrap: wrap;
    gap: 12px;
  }
  
  .file-name {
    order: 3;
    width: 100%;
    max-width: 100%;
    text-align: center;
  }
  
  .label-section {
    flex-direction: column;
    gap: 16px;
    padding: 16px;
  }
  
  .score-group {
    max-width: 100%;
    width: 100%;
  }
  
  .save-btn {
    width: 100%;
  }
}
</style>
