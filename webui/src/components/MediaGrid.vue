<template>
  <div class="media-grid">
    <div v-if="!mediaFiles.length" class="empty-state">
      <p>{{ emptyMessage }}</p>
    </div>
    <div v-else class="grid-container">
      <div
        v-for="(file, index) in paginatedFiles"
        :key="file.path"
        class="media-item"
        :class="{ selected: selectedFile?.path === file.path, labeled: hasLabel(file.path) }"
        @click="handleItemClick(file)" @dblclick="handleItemDoubleClick(file)"
      >
        <div class="thumbnail">
          <div v-if="file.inference_score != null" class="inference-score-badge">
            {{ formatScore(file.inference_score) }}
          </div>
          <div v-if="isImage(file.name) || isVideo(file.name)" class="thumbnail-skeleton" :class="{ 'skeleton-loaded': loadingStates[file.path] === 'loaded' }"></div>
          <img
            v-if="isImage(file.name)"
            :src="getThumbnailUrl(file.path)"
            :alt="file.name"
            @error="handleImageError"
            @load="onImageLoad(file.path)"
            loading="lazy"
            :class="{ 'loading': loadingStates[file.path] !== 'loaded', 'loaded': loadingStates[file.path] === 'loaded' }"
          />
          <div v-else-if="isVideo(file.name)" class="video-thumbnail">
            <!-- Show stitched screenshot image if available -->
            <img
              v-if="screenshotsData[file.path]?.status === 'cached' && screenshotsData[file.path]?.screenshot_url"
              :src="screenshotsData[file.path].screenshot_url"
              :alt="file.name"
              @error="onScreenshotError($event, file.path)"
              @load="onScreenshotLoad(file.path)"
              loading="lazy"
              class="stitched-screenshot"
              :class="{ 'loaded': loadingStates[file.path] === 'loaded' }"
            />
            <!-- Show loading skeleton while screenshots are being generated -->
            <div v-else-if="screenshotsData[file.path]?.status === 'processing'" class="screenshots-loading">
              <span class="loading-text">生成截图中...</span>
            </div>
            <!-- Fallback to single thumbnail -->
            <img
              v-else
              :src="getThumbnailUrl(file.path)"
              :alt="file.name"
              @error="onVideoThumbnailError($event, file)"
              @load="onImageLoad(file.path)"
              loading="lazy"
              :class="{ 'loading': loadingStates[file.path] !== 'loaded', 'loaded': loadingStates[file.path] === 'loaded' }"
            />
            <div class="video-overlay">
              <span class="video-play-icon">▶</span>
              <span v-if="file.duration" class="video-duration">{{ formatDuration(file.duration) }}</span>
            </div>
          </div>
          <div v-else class="file-placeholder">
            <span class="file-icon">📄</span>
          </div>
        </div>
        <div class="file-info">
          <div class="file-text">
            <span class="file-name">{{ file.name }}</span>
            <span v-if="file.inference_score != null" class="inference-score-text">
              模型分 {{ formatScore(file.inference_score) }}
            </span>
          </div>
          <span v-if="hasLabel(file.path)" class="label-indicator" title="已标注">
            <span class="check-icon">✓</span>
          </span>
        </div>
      </div>
    </div>
    <div v-if="mediaFiles.length > 0" class="pagination">
      <button
        :disabled="currentPage === 1"
        @click="currentPage--"
        class="page-btn"
      >
        上一页
      </button>
      <span class="page-info">{{ currentPage }} / {{ totalPages }}</span>
      <button
        :disabled="currentPage === totalPages"
        @click="currentPage++"
        class="page-btn"
      >
        下一页
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, shallowRef, computed, watch, onUnmounted, toRaw } from 'vue'
import { getThumbnailUrl, getScreenshots, preloadMedia } from '../api/index.js'

const props = defineProps({
  mediaFiles: {
    type: Array,
    default: () => []
  },
  emptyMessage: {
    type: String,
    default: '选择文件夹以加载媒体文件'
  },
  labels: {
    type: Array,
    default: () => []
  },
  itemsPerPage: {
    type: Number,
    default: 20
  }
})

const emit = defineEmits(['select', 'preview'])

const selectedFile = shallowRef(null)
const currentPage = ref(1)
const loadingStates = ref({}) // { [path]: 'loading' | 'loaded' | 'error' }
const screenshotsData = ref({}) // { [path]: { status: 'cached' | 'processing', contentHash?: string, screenshots?: string[] } }
const retryCount = ref({}) // { [path]: number }
const MAX_RETRIES = 2

const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg']
const videoExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm']

const totalPages = computed(() =>
  Math.ceil(props.mediaFiles.length / props.itemsPerPage)
)

const paginatedFiles = computed(() => {
  const start = (currentPage.value - 1) * props.itemsPerPage
  const end = start + props.itemsPerPage
  return props.mediaFiles.slice(start, end)
})

// 使用 Map 缓存 labels，O(1) 查找替代 O(n) 搜索
const labelMap = computed(() => {
  const map = new Map()
  const rawLabels = toRaw(props.labels)
  for (let i = 0; i < rawLabels.length; i++) {
    map.set(rawLabels[i].media_path, rawLabels[i])
  }
  return map
})

function isImage(filename) {
  const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase()
  return imageExtensions.includes(ext)
}

function isVideo(filename) {
  const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase()
  return videoExtensions.includes(ext)
}

function hasLabel(filePath) {
  return labelMap.value.has(filePath)
}

function handleItemClick(file) {
  selectedFile.value = file
  emit('select', file)
}

function handleItemDoubleClick(file) {
  selectedFile.value = file
  emit('select', file)
  emit('preview', file)
}

function handleImageError(event) {
  const img = event.target
  const src = img.src
  
  // Extract path from URL
  const urlParams = new URLSearchParams(src.split('?')[1])
  const path = urlParams.get('path')
  
  if (!path) {
    // Fallback to placeholder if we can't extract path
    img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="50" y="50" text-anchor="middle" font-size="40">🖼️</text></svg>'
    return
  }
  
  // Check retry count
  const currentRetries = retryCount.value[path] || 0
  
  if (currentRetries < MAX_RETRIES) {
    // Trigger preload and retry after delay
    retryCount.value[path] = currentRetries + 1
    loadingStates.value[path] = 'loading'
    
    // Request thumbnail generation
    preloadMedia([path], ['thumbnail']).catch(() => {})
    
    // Retry after 1 second
    setTimeout(() => {
      // Force reload by adding timestamp
      img.src = getThumbnailUrl(path) + '&_t=' + Date.now()
    }, 1000)
  } else {
    // Max retries reached, show placeholder
    img.classList.remove('loaded')
    img.classList.add('loading')
    img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><text x="50" y="50" text-anchor="middle" font-size="40">🖼️</text></svg>'
    loadingStates.value[path] = 'error'
  }
}

function onImageLoad(path) {
  loadingStates.value[path] = 'loaded'
}

function onScreenshotLoad(path) {
  loadingStates.value[path] = 'loaded'
}

function onScreenshotError(event, path) {
  // Fallback to thumbnail on screenshot error
  const img = event.target
  img.src = getThumbnailUrl(path)
}

function onVideoMetadata(event, file) {
  const video = event.target
  if (video && video.duration && !isNaN(video.duration)) {
    file.duration = video.duration
  }
}

function onVideoError(file) {
  // Mark as having no duration on error
  file.duration = null
}

async function onVideoThumbnailError(event, file) {
  const path = file.path
  
  // If screenshots are not being loaded yet, try to load them
  if (!screenshotsData.value[path]) {
    await loadScreenshots(file)
  }
  
  // If screenshots still not available, retry thumbnail
  const currentRetries = retryCount.value[path] || 0
  if (currentRetries < MAX_RETRIES && screenshotsData.value[path]?.status !== 'cached') {
    retryCount.value[path] = currentRetries + 1
    preloadMedia([path], ['thumbnail']).catch(() => {})
    
    const img = event.target
    setTimeout(() => {
      img.src = getThumbnailUrl(path) + '&_t=' + Date.now()
    }, 1000)
  }
}

async function loadScreenshots(file) {
  const path = file.path
  
  // Check if already loaded or processing
  if (screenshotsData.value[path]?.status === 'cached') {
    return // Already loaded, skip
  }
  
  // Mark as processing
  screenshotsData.value[path] = { status: 'processing' }
  
  try {
    const result = await getScreenshots(path)
    if (result.status === 'cached' && result.screenshot_url) {
      screenshotsData.value[path] = {
        status: 'cached',
        screenshot_url: result.screenshot_url,
        contentHash: result.content_hash
      }
      loadingStates.value[path] = 'loaded'
    } else if (result.status === 'processing') {
      // Poll for completion
      pollScreenshots(file, result.task_id)
    }
  } catch (error) {
    console.error('Failed to load screenshots:', error)
    screenshotsData.value[path] = { status: 'error' }
  }
}

async function pollScreenshots(file, taskId, attempts = 0) {
  if (attempts >= 30) { // Max 30 attempts (30 seconds)
    screenshotsData.value[file.path] = { status: 'error' }
    return
  }
  
  await new Promise(resolve => setTimeout(resolve, 1000))
  
  try {
    const result = await getScreenshots(file.path)
    if (result.status === 'cached' && result.screenshot_url) {
      screenshotsData.value[file.path] = {
        status: 'cached',
        screenshot_url: result.screenshot_url,
        contentHash: result.content_hash
      }
      loadingStates.value[file.path] = 'loaded'
    } else {
      // Continue polling
      pollScreenshots(file, taskId, attempts + 1)
    }
  } catch (error) {
    // Continue polling on error
    pollScreenshots(file, taskId, attempts + 1)
  }
}

function formatDuration(seconds) {
  if (!seconds || isNaN(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  if (mins >= 60) {
    const hours = Math.floor(mins / 60)
    const remainingMins = mins % 60
    return `${hours}:${remainingMins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function formatScore(score) {
  return Number(score).toFixed(4).replace(/\.?0+$/, '')
}

// Reset to first page when media files change
watch(() => props.mediaFiles.length, () => {
  currentPage.value = 1
})

// Initialize screenshots for visible video files
let debounceTimer = null
watch(() => paginatedFiles.value, (files) => {
  // Clear previous timer
  if (debounceTimer) clearTimeout(debounceTimer)
  
  // Debounce for 200ms to avoid rapid API calls
  debounceTimer = setTimeout(() => {
    files.forEach(file => {
      // Load screenshots for videos that haven't been loaded yet
      if (isVideo(file.name) && !screenshotsData.value[file.path]) {
        loadScreenshots(file)
      }
    })
  }, 200)
}, { immediate: true })

defineExpose({
  selectedFile,
  selectNext,
  selectPrevious
})

function selectNext() {
  const currentIndex = paginatedFiles.value.findIndex(f => f.path === selectedFile.value?.path)
  if (currentIndex < paginatedFiles.value.length - 1) {
    handleItemClick(paginatedFiles.value[currentIndex + 1])
  } else if (currentPage.value < totalPages.value) {
    currentPage.value++
    handleItemClick(paginatedFiles.value[0])
  }
}

function selectPrevious() {
  const currentIndex = paginatedFiles.value.findIndex(f => f.path === selectedFile.value?.path)
  if (currentIndex > 0) {
    handleItemClick(paginatedFiles.value[currentIndex - 1])
  } else if (currentPage.value > 1) {
    currentPage.value--
    handleItemClick(paginatedFiles.value[paginatedFiles.value.length - 1])
  }
}
</script>

<style scoped>
.media-grid {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.empty-state {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: #888;
  font-size: 14px;
}

.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: 12px;
  padding: 16px;
  overflow-y: auto;
  flex: 1;
}

.media-item {
  display: flex;
  flex-direction: column;
  cursor: pointer;
  border-radius: 8px;
  overflow: hidden;
  background: #2a2a2a;
  border: 2px solid transparent;
  position: relative;
}

.media-item:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}

.media-item.selected {
  border-color: #42b883;
  box-shadow: 0 0 0 3px rgba(66, 184, 131, 0.3);
}

.media-item.labeled {
  border-color: rgba(66, 184, 131, 0.5);
}

.media-item.labeled::after {
  content: '';
  position: absolute;
  top: 6px;
  right: 6px;
  width: 20px;
  height: 20px;
  background: #42b883;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  z-index: 2;
}

.thumbnail {
  aspect-ratio: 1;
  overflow: hidden;
  background: #1a1a1a;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.inference-score-badge {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: 3;
  padding: 4px 7px;
  border-radius: 999px;
  background: rgba(10, 10, 10, 0.82);
  border: 1px solid rgba(66, 184, 131, 0.35);
  color: #7ad7aa;
  font-size: 10px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.thumbnail-skeleton {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    var(--skeleton-start, #f0f0f0) 25%,
    var(--skeleton-mid, #e0e0e0) 50%,
    var(--skeleton-start, #f0f0f0) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 0.8s ease-in-out infinite;
  transition: opacity 0.1s ease;
}

.thumbnail-skeleton.skeleton-loaded {
  opacity: 0;
  pointer-events: none;
}

.thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  position: relative;
  z-index: 1;
}

.thumbnail img.loading {
  opacity: 0;
}

.thumbnail img.loaded {
  opacity: 1;
}

@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

@media (prefers-color-scheme: dark) {
  .thumbnail-skeleton {
    --skeleton-start: #2d2d2d;
    --skeleton-mid: #3d3d3d;
  }
}

.video-thumbnail {
  width: 100%;
  height: 100%;
  position: relative;
  background: #1a1a1a;
  overflow: hidden;
}

.video-thumbnail video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.stitched-screenshot {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.screenshots-loading {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1a1a1a;
}

.loading-text {
  color: #888;
  font-size: 10px;
}

.video-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.3) 0%, rgba(0, 0, 0, 0.6) 100%);
  transition: background 0.08s ease;
  z-index: 10;
}

.media-item:hover .video-overlay {
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.2) 0%, rgba(0, 0, 0, 0.5) 100%);
}

.video-play-icon {
  font-size: 28px;
  color: #fff;
  opacity: 0.9;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
}

.media-item:hover .video-play-icon {
  opacity: 1;
}

.video-duration {
  position: absolute;
  bottom: 6px;
  right: 6px;
  background: rgba(0, 0, 0, 0.8);
  color: #fff;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

.file-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #333 0%, #222 100%);
}

.file-icon {
  font-size: 32px;
  opacity: 0.7;
}

.file-info {
  padding: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.file-text {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.file-name {
  font-size: 11px;
  color: #ccc;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.inference-score-text {
  font-size: 10px;
  color: #7ad7aa;
  font-variant-numeric: tabular-nums;
}

.label-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
}

.check-icon {
  color: #42b883;
  font-size: 10px;
  font-weight: bold;
  background: rgba(66, 184, 131, 0.15);
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}

.pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 12px;
  border-top: 1px solid #333;
  background: #1a1a1a;
}

.page-btn {
  padding: 6px 16px;
  background: #333;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.08s;
}

.page-btn:hover:not(:disabled) {
  background: #444;
}

.page-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.page-info {
  font-size: 13px;
  color: #888;
  min-width: 60px;
  text-align: center;
}
</style>
