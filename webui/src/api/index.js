import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000,
})

/**
 * Get backend health status
 * @returns {Promise<Object>} Health check response
 */
export async function getHealth() {
  const response = await api.get('/health')
  return response.data
}

/**
 * Get detailed backend status
 * @returns {Promise<Object>} Status information
 */
export async function getStatus() {
  const response = await api.get('/status')
  return response.data
}

/**
 * Score a single media file
 * @param {File} file - Media file to score
 * @param {boolean} isVideo - Whether the file is a video
 * @returns {Promise<Object>} Score result
 */
export async function scoreMedia(file, isVideo) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('is_video', isVideo)

  const response = await api.post('/score', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return response.data
}

/**
 * Score multiple media items in batch
 * @param {Array<{path: string, is_video: boolean}>} items - Array of items to score
 * @returns {Promise<Object>} Batch score results
 */
export async function batchScore(items) {
  const response = await api.post('/batch-score', { items })
  return response.data
}

/**
 * Get labels with optional filtering
 * @param {number} [limit] - Limit number of results (optional)
 * @returns {Promise<Object>} Labels list
 */
export async function getLabels(limit) {
  const params = {}
  if (limit) params.limit = limit

  const response = await api.get('/labels', { params })
  return response.data
}

/**
 * Create a new label
 * @param {string} mediaPath - Path to media file
 * @param {number} score - Score value
 * @returns {Promise<Object>} Created label
 */
export async function createLabel(mediaPath, score) {
  const response = await api.post('/label', { media_path: mediaPath, score })
  return response.data
}

/**
 * Delete a label by ID
 * @param {number} labelId - ID of the label to delete
 * @returns {Promise<Object>} Deletion result
 */
export async function deleteLabel(labelId) {
  const response = await api.delete(`/label/${labelId}`)
  return response.data
}

/**
 * Export labels dataset
 * @param {string} [format] - Export format: 'json' or 'csv'
 * @returns {Promise<Blob>} Export file blob
 */
export async function exportLabels(format = 'json') {
  const params = {}
  if (format) params.format = format

  const response = await api.get('/export/labels', {
    params,
    responseType: 'blob'
  })
  return response.data
}

/**
 * Get export statistics
 * @returns {Promise<Object>} Export stats with total
 */
export async function getExportStats() {
  const response = await api.get('/export/stats')
  return response.data
}

/**
 * Scan server folder for media files
 * @param {string} path - Folder path to scan
 * @param {boolean} recursive - Whether to scan recursively
 * @returns {Promise<{files: Array<{path: string, name: string, type: string}>}>}
 */
export async function scanFolder(path, recursive = false) {
  const response = await api.get('/media/scan', { 
    params: { path, recursive } 
  })
  return response.data
}

/**
 * Load completed inferred media from database, sorted by score descending
 * @param {Object} options
 * @param {number} [options.limit] - Maximum number of results to return
 * @param {string} [options.rootPath] - Optional root path filter
 * @returns {Promise<{files: Array<{path: string, name: string, type: string, inference_score: number}>}>}
 */
export async function getInferredMedia({ limit = 50000, rootPath } = {}) {
  const params = { limit }
  if (rootPath) params.root_path = rootPath

  const response = await api.get('/media/inference', {
    params,
    timeout: 30000,
  })
  return response.data
}

/**
 * Preload media files (thumbnails and/or transcodes)
 * @param {string[]} paths - Array of file paths to preload
 * @param {string[]} types - Types to preload: ['thumbnail', 'transcode']
 * @returns {Promise<{submitted: number, cached: number, total: number}>}
 */
export async function preloadMedia(paths, types = ['thumbnail']) {
  const response = await api.post('/media/preload', { paths, types })
  return response.data
}

/**
 * Get thumbnail URL for media file (non-blocking, returns image or triggers async generation)
 * @param {string} mediaPath - Path to media file
 * @returns {string} Thumbnail URL
 */
export function getThumbnailUrl(mediaPath) {
  // Use non-blocking endpoint - returns cached image immediately or starts async generation
  return `/api/media/thumbnail?path=${encodeURIComponent(mediaPath)}`
}

/**
 * Get thumbnail URL with wait (blocking, use sparingly)
 * @param {string} mediaPath - Path to media file
 * @param {number} [timeout] - Timeout in seconds (default: 30)
 * @returns {string} Thumbnail URL
 */
export function getThumbnailUrlWait(mediaPath, timeout = 30) {
  return `/api/media/thumbnail/wait?path=${encodeURIComponent(mediaPath)}&timeout=${timeout}`
}

/**
 * Get stream URL for media file (non-blocking for original, async for transcoded)
 * @param {string} mediaPath - Path to media file
 * @param {string} [format] - Output format: 'original', 'webm', or 'mp4'
 * @returns {string} Stream URL
 */
export function getStreamUrl(mediaPath, format = 'original') {
  const params = new URLSearchParams({ path: mediaPath })
  if (format === 'original') {
    return `/api/media/stream?${params.toString()}`
  } else {
    // Use non-blocking endpoint - returns cached video or starts async transcoding
    params.append('format', format)
    return `/api/media/stream?${params.toString()}`
  }
}

/**
 * Get stream URL with wait (blocking, use only for preview modal)
 * @param {string} mediaPath - Path to media file
 * @param {string} [format] - Output format: 'webm' or 'mp4'
 * @param {number} [timeout] - Timeout in seconds (default: 300)
 * @returns {string} Stream URL
 */
export function getStreamUrlWait(mediaPath, format = 'webm', timeout = 300) {
  const params = new URLSearchParams({ path: mediaPath, format, timeout: timeout.toString() })
  return `/api/media/stream/wait?${params.toString()}`
}

/**
 * Get transcode task status
 * @param {string} path - Media file path
 * @returns {Promise<{status: string, progress?: number}>} Task status
 */
export async function getTranscodeStatus(path) {
  const taskId = `transcode_${path}`
  try {
    const response = await api.get(`/media/task/${encodeURIComponent(taskId)}`)
    return response.data
  } catch (error) {
    // Task doesn't exist yet - return pending status
    if (error.response?.status === 404) {
      return { status: 'pending' }
    }
    throw error
  }
}

/**
 * Get screenshots for video file (stitched 5x5 image)
 * @param {string} mediaPath - Path to video file
 * @returns {Promise<{status: string, screenshot_url?: string, content_hash?: string, task_id?: string}>}
 */
export async function getScreenshots(mediaPath) {
  const response = await api.get('/media/screenshots', {
    params: { path: mediaPath }
  })
  return response.data
}

/**
 * Get screenshot URL by content hash (single stitched image)
 * @param {string} contentHash - Content hash of the video
 * @returns {string} Screenshot URL
 */
export function getScreenshotUrl(contentHash) {
  return `/api/media/screenshot/${contentHash}`
}


/**
 * List available server directories
 * @returns {Promise<{directories: Array<{path: string, name: string}>}>}
 */
export async function listDirectories() {
  const response = await api.get('/media/directories')
  return response.data
}

/**
 * Get live stream URL for real-time transcoding
 * @param {string} mediaPath - Path to video file
 * @param {number} startTime - Start time in seconds (default: 0)
 * @param {number} duration - Duration in seconds (default: 30)
 * @returns {string} Live stream URL
 */
export function getLiveStreamUrl(mediaPath, startTime = 0, duration = 30) {
  const params = new URLSearchParams({ 
    path: mediaPath, 
    start_time: startTime.toString(), 
    duration: duration.toString() 
  })
  return `/api/media/stream/live?${params.toString()}`
}

/**
 * Get HLS playlist URL for video streaming
 * @param {string} mediaPath - Path to video file
 * @returns {string} HLS playlist URL (m3u8)
 */
export function getHlsPlaylistUrl(mediaPath) {
  return `/api/media/hls/playlist?path=${encodeURIComponent(mediaPath)}`
}

/**
 * List pipeline jobs.
 * @param {boolean} [includeLogs=false] - Whether to include tail logs.
 * @returns {Promise<{jobs: Array}>}
 */
export async function getPipelineJobs(includeLogs = false) {
  const response = await api.get('/pipeline/jobs', { params: { include_logs: includeLogs } })
  return response.data
}

/**
 * Start a training job from labels and video dataset.
 * @param {Object} payload
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function startPipelineTrain(payload) {
  const response = await api.post('/pipeline/jobs/train', payload)
  return response.data
}

/**
 * Start Telegram gated download job only.
 * @param {Object} payload
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function startPipelineDownload(payload) {
  const response = await api.post('/pipeline/jobs/download', payload)
  return response.data
}

/**
 * Start full Telegram pipeline job: download + optional inference/rebucket.
 * @param {Object} payload
 * @returns {Promise<{job_id: string, status: string}>}
 */
export async function startPipelineGlobal(payload) {
  const response = await api.post('/pipeline/jobs/global-pipeline', payload)
  return response.data
}

/**
 * Stop pipeline job by id.
 * @param {string} jobId
 * @returns {Promise<{status: string}>}
 */
export async function stopPipelineJob(jobId) {
  const response = await api.delete(`/pipeline/jobs/${jobId}`)
  return response.data
}

/**
 * Get active deploy checkpoint.
 * @returns {Promise<{model_checkpoint: string, telegram_checkpoint: string}>}
 */
export async function getDeployState() {
  const response = await api.get('/pipeline/deploy')
  return response.data
}

/**
 * Update checkpoint used by train and downloader stages.
 * @param {string} checkpoint
 * @returns {Promise<{message: string, model_checkpoint: string, telegram_checkpoint: string}>}
 */
export async function setDeployState(checkpoint) {
  const response = await api.post('/pipeline/deploy', { checkpoint })
  return response.data
}
