<template>
  <div class="label-view">
    <div class="main-container">
      <!-- Left Sidebar: Folder Browser -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <h2>标注数据源</h2>
        </div>

        <div class="source-switch">
          <button
            class="source-btn"
            :class="{ active: dataSource === SOURCE_FOLDER }"
            :disabled="isLoading"
            @click="switchDataSource(SOURCE_FOLDER)"
          >
            文件夹扫描
          </button>
          <button
            class="source-btn"
            :class="{ active: dataSource === SOURCE_INFERENCE }"
            :disabled="isLoading"
            @click="switchDataSource(SOURCE_INFERENCE)"
          >
            已推理结果
          </button>
        </div>
        
        <div v-if="dataSource === SOURCE_FOLDER" class="folder-input-section">
          <label for="folder-path">文件夹路径</label>
          <input
            id="folder-path"
            v-model="folderPath"
            type="text"
            placeholder="输入文件夹路径，如 /path/to/media"
            class="folder-input"
            @keyup.enter="loadFolder"
          />
          <div class="folder-actions">
            <label class="recursive-checkbox">
              <input type="checkbox" v-model="recursive" />
              <span>递归</span>
            </label>
            <button
              @click="loadFolder"
              :disabled="isLoading || !folderPath"
              class="folder-btn primary"
            >
              {{ isLoading ? '扫描中...' : '扫描' }}
            </button>
          </div>
        </div>

        <div v-else class="inference-mode-note">
          <div class="note-title">已推理结果模式</div>
          <div class="note-body">
            直接使用数据库中已完成推理的媒体，按模型分数从高到低展示。
          </div>
        </div>
        
        <div class="status-bar">
          <div class="status-item">
            <span class="status-label">总文件数:</span>
            <span class="status-value">{{ mediaFiles.length }}</span>
          </div>
          <div class="status-item">
            <span class="status-label">已标注:</span>
            <span class="status-value labeled">{{ labels.length }}</span>
          </div>
          <div class="status-item">
            <span class="status-label">待标注:</span>
            <span class="status-value pending">{{ mediaFiles.length - labeledCount }}</span>
          </div>
        </div>
        
        <div v-if="isLoading" class="loading-indicator">
          <div class="spinner"></div>
          <span>{{ loadingMessage }}</span>
        </div>
        
        <div v-if="error" class="error-message">
          {{ error }}
        </div>
        
        <!-- Export Section -->
        <div class="export-section">
          <h3>导出数据集</h3>
          
          <div class="export-stats">
            <div class="export-stat-item">
              <span class="stat-label">总标签:</span>
              <span class="stat-value">{{ exportStats.total }}</span>
            </div>
          </div>
          
          <div class="export-controls">
            <select v-model="exportFormat" class="export-select">
              <option value="json">JSON</option>
              <option value="csv">CSV</option>
            </select>
            
            <button
              @click="handleExport"
              :disabled="isExporting || exportStats.total === 0"
              class="export-btn"
            >
              {{ isExporting ? '导出中...' : '导出' }}
            </button>
          </div>
        </div>
      </aside>
      
      <!-- Main Area: Media Grid -->
      <main class="media-area">
        <div class="media-header">
          <div class="header-title">
            <h1>媒体标注</h1>
            <span class="source-chip">
              {{ dataSource === SOURCE_INFERENCE ? '数据库已推理结果' : '文件夹扫描结果' }}
            </span>
          </div>
          <div class="header-actions">
            <button
              @click="refreshLabels"
              :disabled="isRefreshing"
              class="refresh-btn"
            >
              {{ isRefreshing ? '刷新中...' : '刷新标签' }}
            </button>
          </div>
        </div>
        
        <div class="grid-wrapper">
          <MediaGrid
            ref="mediaGridRef"
            :media-files="mediaFiles"
            :labels="labels"
            :empty-message="emptyStateMessage"
            :items-per-page="20"
            @select="handleFileSelect"
            @preview="openPreview"
          />
        </div>
      </main>
    </div>
    
    <!-- Bottom: Label Panel -->
    <div class="label-section">
      <LabelPanel
        :selected-file="selectedFile"
        :labels="labels"
        :is-preview-open="isPreviewOpen"
        @save="handleSaveLabel"
        @next="handleNextFile"
        @previous="handlePreviousFile"
      />
    </div>

    <!-- Media Preview Modal -->
    <MediaPreview
      :is-open="isPreviewOpen"
      :files="mediaFiles"
      :current-index="previewIndex"
      :labels="labels"
      @close="closePreview"
      @navigate="handlePreviewNavigate"
      @save="handlePreviewSave"
    />
  </div>
</template>

<script setup>
import { ref, computed, shallowRef, onMounted, markRaw } from 'vue'
import MediaGrid from '../components/MediaGrid.vue'
import LabelPanel from '../components/LabelPanel.vue'
import MediaPreview from '../components/MediaPreview.vue'
import { getLabels, createLabel, getExportStats, exportLabels, scanFolder as apiScanFolder, getInferredMedia, getThumbnailUrl, preloadMedia } from '../api/index.js'

const SOURCE_FOLDER = 'folder'
const SOURCE_INFERENCE = 'inference'

// State
const folderPath = ref('')
const recursive = ref(false)
const dataSource = ref(SOURCE_FOLDER)
const folderMediaFiles = shallowRef([])
const inferredMediaFiles = shallowRef([])
const labels = shallowRef([])
const selectedFile = shallowRef(null)
const isLoading = ref(false)
const isRefreshing = ref(false)
const scannedCount = ref(0)
const error = ref(null)
const mediaGridRef = ref(null)

// Export state
const exportStats = ref({ total: 0 })
const exportFormat = ref('json')
const isExporting = ref(false)

// Preview state
const isPreviewOpen = ref(false)
const previewIndex = ref(0)

const mediaFiles = computed(() => (
  dataSource.value === SOURCE_INFERENCE
    ? inferredMediaFiles.value
    : folderMediaFiles.value
))

const loadingMessage = computed(() => {
  if (dataSource.value === SOURCE_INFERENCE) {
    return '正在加载数据库中的已推理媒体...'
  }
  return `正在扫描文件...${scannedCount.value} 个`
})

const emptyStateMessage = computed(() => (
  dataSource.value === SOURCE_INFERENCE
    ? '点击“已推理结果”后，会从数据库加载已完成推理的媒体'
    : '选择文件夹以加载媒体文件'
))

// Cache media paths for fast lookup
const mediaPathsSet = computed(() => new Set(mediaFiles.value.map(f => f.path)))

const labeledCount = computed(() => {
  return labels.value.filter(l => mediaPathsSet.value.has(l.media_path)).length
})

// Smart preload constants
const PAGE_SIZE = 20  // Visible items per page
const LARGE_FOLDER_THRESHOLD = 50
const BATCH_SIZE = 10  // Reduced for smoother loading
const BATCH_DELAY_MS = 300  // Faster batch interval

// Media extension constants
// Media extension constants
const videoExts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
const imageExts = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff']

function isVideo(filename) {
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return videoExts.includes(ext)
}


function isImage(filename) {
  const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'))
  return imageExts.includes(ext)
}

// Smart preload strategy: prioritize visible area, videos, then batch process
function smartPreload(files, enableBackground = true) {
  if (files.length === 0) return

  // Separate videos and images
  const videos = files.filter(f => isVideo(f.name))
  const images = files.filter(f => isImage(f.name))

  // Priority 1: First page thumbnails (visible area) - all types
  const firstPageFiles = files.slice(0, PAGE_SIZE)
  const firstPagePaths = firstPageFiles.map(f => f.path)
  preloadMedia(firstPagePaths, ['thumbnail']).catch(() => {})

  // Priority 2: Videos - only preload thumbnails (no transcode until preview)
  // Transcoding now happens on-demand when user clicks preview

  if (!enableBackground) {
    return
  }

  // Priority 3: Batch preload for large folders (non-blocking)
  if (files.length > LARGE_FOLDER_THRESHOLD) {
    // Schedule background batches with delay to avoid overwhelming
    setTimeout(() => {
      preloadBatches(videos, images)
    }, BATCH_DELAY_MS)
  } else {
    // Small folder: preload remaining thumbnails
    const remainingFiles = files.slice(PAGE_SIZE)
    if (remainingFiles.length > 0) {
      preloadMedia(remainingFiles.map(f => f.path), ['thumbnail']).catch(() => {})
    }
  }
}

// Batch preload remaining files in background
function preloadBatches(videos, images) {
  // Combine all files for thumbnail batches (no transcode priority anymore)
  const remainingFiles = [...videos, ...images]
  
  // Process in batches
  let batchIndex = 0
  const processBatch = () => {
    const start = batchIndex * BATCH_SIZE
    const batch = remainingFiles.slice(start, start + BATCH_SIZE)
    
    if (batch.length > 0) {
      preloadMedia(batch.map(f => f.path), ['thumbnail']).catch(() => {})
      batchIndex++
      
      // Schedule next batch
      if (start + BATCH_SIZE < remainingFiles.length) {
        setTimeout(processBatch, BATCH_DELAY_MS)
      }
    }
  }
  
  // Start first batch after initial delay
  setTimeout(processBatch, BATCH_DELAY_MS)
}
// Load folder from text input (scanning via backend)
async function loadFolder() {
  if (!folderPath.value || isLoading.value) return
  
  dataSource.value = SOURCE_FOLDER
  isLoading.value = true
  error.value = null
  
  try {
    const result = await apiScanFolder(folderPath.value, recursive.value)
    folderMediaFiles.value = markRaw(result.files.map(f => ({
      ...f,
      url: getThumbnailUrl(f.path)
    })))
    // Smart preload strategy
    smartPreload(result.files)
    selectedFile.value = null

    refreshLabels()
    fetchExportStats()
  } catch (err) {
    error.value = '扫描失败：' + (err.message || '未知错误')
  } finally {
    isLoading.value = false
  }
}

async function loadInferenceMedia() {
  if (isLoading.value) return

  isLoading.value = true
  error.value = null

  try {
    const result = await getInferredMedia()
    const files = result.files.map(file => ({
      ...file,
      url: getThumbnailUrl(file.path)
    }))
    inferredMediaFiles.value = markRaw(files)
    selectedFile.value = null

    smartPreload(files, false)
    await refreshLabels()
    await fetchExportStats()
  } catch (err) {
    error.value = '加载已推理结果失败：' + (err.message || '未知错误')
  } finally {
    isLoading.value = false
  }
}

async function switchDataSource(source) {
  if (source === dataSource.value && mediaFiles.value.length > 0) return

  dataSource.value = source
  selectedFile.value = null
  error.value = null

  if (source === SOURCE_INFERENCE && inferredMediaFiles.value.length === 0) {
    await loadInferenceMedia()
  }
}


// Handle file selection from grid
function handleFileSelect(file) {
  selectedFile.value = file
}

function updateLocalLabel(mediaPath, score) {
  const existingIndex = labels.value.findIndex(l => l.media_path === mediaPath)
  
  if (existingIndex >= 0) {
    const newLabels = [...labels.value]
    newLabels[existingIndex] = { ...newLabels[existingIndex], score }
    labels.value = markRaw(newLabels)
  } else {
    labels.value = markRaw([...labels.value, {
      id: Date.now(),
      media_path: mediaPath,
      score,
      created_at: new Date().toISOString()
    }])
  }
}

// Save a new label
async function handleSaveLabel({ mediaPath, score }) {
  try {
    await createLabel(mediaPath, score)
    updateLocalLabel(mediaPath, score)
    
    // Refresh export stats
    fetchExportStats()
    
    // Auto-advance to next file
    handleNextFile()
  } catch (err) {
    error.value = '保存失败：' + (err.message || '未知错误')
    setTimeout(() => {
      error.value = null
    }, 3000)
  }
}

// Navigate to next file
function handleNextFile() {
  mediaGridRef.value?.selectNext()
}

// Navigate to previous file
function handlePreviousFile() {
  mediaGridRef.value?.selectPrevious()
}

// Preview handlers
function openPreview(file) {
  const index = mediaFiles.value.findIndex(f => f.path === file.path)
  if (index !== -1) {
    previewIndex.value = index
    isPreviewOpen.value = true
  }
}

function closePreview() {
  isPreviewOpen.value = false
}

function handlePreviewNavigate(index) {
  previewIndex.value = index
  // Update selected file to match preview
  const file = mediaFiles.value[index]
  if (file) {
    selectedFile.value = file
  }
}

async function handlePreviewSave({ mediaPath, score }) {
  try {
    await createLabel(mediaPath, score)
    updateLocalLabel(mediaPath, score)

    // Refresh export stats
    fetchExportStats()
  } catch (err) {
    error.value = '保存失败：' + (err.message || '未知错误')
    setTimeout(() => {
      error.value = null
    }, 3000)
  }
}

// Refresh labels from backend
async function refreshLabels() {
  if (isRefreshing.value) return
  
  isRefreshing.value = true
  try {
    const data = await getLabels(50000)
    const nextLabels = Array.isArray(data) ? data : (data.labels || [])
    labels.value = markRaw(nextLabels)
  } catch (err) {
    console.error('Failed to load labels:', err)
  } finally {
    isRefreshing.value = false
  }
}

// Fetch export stats
async function fetchExportStats() {
  try {
    const data = await getExportStats()
    exportStats.value = {
      total: data.total ?? data.total_labels ?? 0,
      labeled_files: data.labeled_files ?? 0
    }
  } catch (err) {
    console.error('Failed to load export stats:', err)
  }
}

// Handle export
async function handleExport() {
  if (isExporting.value || exportStats.value.total === 0) return
  
  isExporting.value = true
  try {
    const blob = await exportLabels(exportFormat.value)
    
    // Create download link
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    
    // Set filename
    a.download = `labels.${exportFormat.value}`
    
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  } catch (err) {
    error.value = '导出失败：' + (err.message || '未知错误')
    setTimeout(() => {
      error.value = null
    }, 3000)
  } finally {
    isExporting.value = false
  }
}

// Load labels on mount
onMounted(() => {
  refreshLabels()
  fetchExportStats()
})
</script>

<style scoped>
.label-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #121212;
  color: #fff;
}

.main-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Sidebar Styles */
.sidebar {
  width: 280px;
  background: #1a1a1a;
  border-right: 1px solid #333;
  display: flex;
  flex-direction: column;
  padding: 20px;
  gap: 20px;
}

.sidebar-header h2 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.source-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.source-btn {
  padding: 10px 12px;
  border: 1px solid #444;
  border-radius: 8px;
  background: #252525;
  color: #aaa;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.08s;
}

.source-btn:hover:not(:disabled) {
  border-color: #42b883;
  color: #fff;
}

.source-btn.active {
  background: rgba(66, 184, 131, 0.16);
  border-color: #42b883;
  color: #42b883;
}

.source-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.folder-input-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.inference-mode-note {
  background: #252525;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.note-title {
  font-size: 13px;
  font-weight: 600;
  color: #fff;
}

.note-body {
  font-size: 12px;
  line-height: 1.5;
  color: #9aa0a6;
}

.folder-input-section label {
  font-size: 12px;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-weight: 500;
}

.folder-input {
  padding: 10px 12px;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #fff;
  font-size: 13px;
  outline: none;
  transition: border-color 0.08s;
}

.folder-input:focus {
  border-color: #42b883;
}

.folder-input::placeholder {
  color: #666;
}

.folder-actions {
  display: flex;
  gap: 8px;
}

.folder-btn {
  flex: 1;
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.08s;
}

.folder-btn.primary {
  background: #42b883;
  color: #fff;
}

.folder-btn.primary:hover:not(:disabled) {
  background: #369e6e;
}

.folder-btn.primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.status-bar {
  background: #252525;
  border-radius: 8px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}

.status-label {
  color: #888;
}

.status-value {
  font-weight: 600;
  color: #fff;
}

.status-value.labeled {
  color: #42b883;
}

.status-value.pending {
  color: #f0a020;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: rgba(66, 184, 131, 0.1);
  border-radius: 6px;
  font-size: 13px;
  color: #42b883;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(66, 184, 131, 0.3);
  border-top-color: #42b883;
  border-radius: 50%;
  animation: spin 0.5s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.error-message {
  padding: 12px;
  background: rgba(220, 53, 69, 0.1);
  border: 1px solid rgba(220, 53, 69, 0.3);
  border-radius: 6px;
  color: #dc3545;
  font-size: 13px;
}

/* Main Media Area */
.media-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.media-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid #333;
  background: #1a1a1a;
}

.header-title {
  display: flex;
  align-items: center;
  gap: 12px;
}

.media-header h1 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.source-chip {
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(66, 184, 131, 0.12);
  color: #7ad7aa;
  font-size: 12px;
  font-weight: 600;
}

.refresh-btn {
  padding: 8px 16px;
  background: #333;
  border: none;
  border-radius: 6px;
  color: #ccc;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.08s;
}

.refresh-btn:hover:not(:disabled) {
  background: #444;
  color: #fff;
}

.refresh-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.grid-wrapper {
  flex: 1;
  overflow: hidden;
}

/* Label Section */
.label-section {
  flex-shrink: 0;
  min-height: 200px;
  max-height: 300px;
  overflow: visible;
}

/* Export Section */
.export-section {
  margin-top: auto;
  background: #252525;
  border-radius: 8px;
  padding: 16px;
}

.export-section h3 {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: #fff;
}

.export-stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

.export-stat-item {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
}

.export-stat-item .stat-label {
  color: #888;
}

.export-stat-item .stat-value {
  color: #42b883;
  font-weight: 600;
}

.export-controls {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.export-select {
  padding: 8px 10px;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 6px;
  color: #fff;
  font-size: 12px;
  outline: none;
  cursor: pointer;
}

.export-select:focus {
  border-color: #42b883;
}

.export-btn {
  padding: 10px 16px;
  background: #42b883;
  border: none;
  border-radius: 6px;
  color: #fff;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.08s;
}

.export-btn:hover:not(:disabled) {
  background: #369e6e;
}

.export-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
