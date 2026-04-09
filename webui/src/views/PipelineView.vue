<template>
  <div class="pipeline-view">
    <header class="pipeline-header">
      <h1>端到端 Pipeline</h1>
      <p>标注 -> 数据导出 -> 训练 -> 部署 -> 下载器</p>
      <p class="header-link">
        <RouterLink to="/label">去标注页</RouterLink>
      </p>
    </header>

    <section class="pipeline-section">
      <h2>部署版本</h2>
      <p class="muted">
        当前模型: {{ deployState.model_checkpoint || '未配置' }}<br />
        当前下载器模型: {{ deployState.telegram_checkpoint || '未配置' }}
      </p>
      <div class="inline-form">
        <label for="checkpoint">统一checkpoint</label>
        <input
          id="checkpoint"
          v-model="deployInput"
          placeholder="例如 outputs/.../checkpoint_best.pt"
          @keyup.enter="updateDeployState"
        />
        <button :disabled="isUpdatingDeploy" @click="updateDeployState">
          {{ isUpdatingDeploy ? '更新中...' : '更新模型' }}
        </button>
      </div>
      <p v-if="deployMessage" class="message">{{ deployMessage }}</p>
    </section>

    <section class="pipeline-section">
      <h2>数据导出</h2>
      <div class="inline-form">
        <button :disabled="isExporting" @click="exportData">
          {{ isExporting ? '导出中...' : '导出labels.json' }}
        </button>
        <span class="muted">已标注: {{ labelTotal }}</span>
      </div>
      <p v-if="exportError" class="error">{{ exportError }}</p>
    </section>

    <section class="pipeline-section">
      <h2>训练（Frozen CLIP）</h2>
      <form class="form-grid" @submit.prevent="startTrain">
        <label>
          labels_path
          <input v-model="trainForm.labelsPath" />
        </label>
        <label>
          val_ratio
          <input v-model.number="trainForm.valRatio" type="number" min="0" max="1" step="0.05" />
        </label>
        <label>
          num_frames
          <input v-model.number="trainForm.numFrames" type="number" min="1" />
        </label>
        <label>
          long_video_strategy
          <select v-model="trainForm.longVideoStrategy">
            <option value="expand">expand</option>
            <option value="compress">compress</option>
          </select>
        </label>
        <label>
          epochs
          <input v-model.number="trainForm.epochs" type="number" min="1" />
        </label>
        <label>
          batch_size
          <input v-model.number="trainForm.batchSize" type="number" min="1" />
        </label>
        <label>
          learning_rate
          <input v-model.number="trainForm.learningRate" type="number" min="0" step="0.0001" />
        </label>
        <label>
          precision
          <select v-model="trainForm.precision">
            <option value="fp32">fp32</option>
            <option value="fp16">fp16</option>
            <option value="bf16">bf16</option>
          </select>
        </label>
        <label>
          output_dir
          <input v-model="trainForm.outputDir" />
        </label>
        <label>
          score_min
          <input v-model="trainForm.scoreMin" placeholder="留空表示自动推断" />
        </label>
        <label>
          score_max
          <input v-model="trainForm.scoreMax" placeholder="留空表示自动推断" />
        </label>
        <label>
          clip_batch_size
          <input v-model.number="trainForm.clipBatchSize" type="number" min="1" />
        </label>
        <label>
          max_frames
          <input v-model.number="trainForm.maxFrames" type="number" min="1" />
        </label>
        <label>
          <input v-model="trainForm.persistentWorkers" type="checkbox" />
          persistent_workers
        </label>
        <div class="form-actions">
          <button type="submit" :disabled="isStartingTrain">
            {{ isStartingTrain ? '启动中...' : '启动训练' }}
          </button>
        </div>
      </form>
      <p v-if="trainError" class="error">{{ trainError }}</p>
    </section>

    <section class="pipeline-section">
      <h2>下载器仅下载阶段</h2>
      <form class="form-grid" @submit.prevent="startDownload">
        <label>
          min_score
          <input v-model.number="downloadForm.minScore" type="number" step="0.1" />
        </label>
        <label>
          chat_batch_size
          <input v-model.number="downloadForm.chatBatchSize" type="number" min="1" />
        </label>
        <label>
          breadth_rounds
          <input v-model.number="downloadForm.breadthRounds" type="number" min="1" />
        </label>
        <label>
          continuous
          <input v-model="downloadForm.continuous" type="checkbox" />
        </label>
        <label>
          keep_below_threshold
          <input v-model="downloadForm.keepBelowThreshold" type="checkbox" />
        </label>
        <label>
          session_name
          <input v-model="downloadForm.sessionName" />
        </label>
        <label>
          discover_chat_types
          <input v-model="downloadForm.discoverChatTypes" placeholder="channel,supergroup,group,private" />
        </label>
        <div class="form-actions">
          <button type="submit" :disabled="isStartingDownload">
            {{ isStartingDownload ? '启动中...' : '启动下载' }}
          </button>
        </div>
      </form>
      <p v-if="downloadError" class="error">{{ downloadError }}</p>
    </section>

    <section class="pipeline-section">
      <h2>全链路（下载 + 可选重算 + 阈值归档）</h2>
      <form class="form-grid" @submit.prevent="startGlobal">
        <label>
          min_score
          <input v-model.number="globalForm.minScore" type="number" step="0.1" />
        </label>
        <label>
          chat_batch_size
          <input v-model.number="globalForm.chatBatchSize" type="number" min="1" />
        </label>
        <label>
          breadth_rounds
          <input v-model.number="globalForm.breadthRounds" type="number" min="1" />
        </label>
        <label>
          continuous
          <input v-model="globalForm.continuous" type="checkbox" />
        </label>
        <label>
          run_bulk_infer
          <input v-model="globalForm.runBulkInfer" type="checkbox" />
        </label>
        <label>
          skip_rebucket
          <input v-model="globalForm.skipRebucket" type="checkbox" />
        </label>
        <label>
          prune_below_threshold
          <input v-model="globalForm.pruneBelowThreshold" type="checkbox" />
        </label>
        <label>
          dry_run
          <input v-model="globalForm.dryRun" type="checkbox" />
        </label>
        <label>
          rebucket_mode
          <select v-model="globalForm.rebucketMode">
            <option value="symlink">symlink</option>
            <option value="move">move</option>
            <option value="hardlink">hardlink</option>
            <option value="copy">copy</option>
          </select>
        </label>
        <div class="form-actions">
          <button type="submit" :disabled="isStartingGlobal">
            {{ isStartingGlobal ? '启动中...' : '启动全链路' }}
          </button>
        </div>
      </form>
      <p v-if="globalError" class="error">{{ globalError }}</p>
    </section>

    <section class="pipeline-section">
      <h2>任务与日志</h2>
      <div class="jobs-header">
        <button @click="refreshJobs" :disabled="jobsLoading">
          {{ jobsLoading ? '刷新中...' : '刷新任务' }}
        </button>
      </div>
      <div v-if="pipelineJobs.length === 0" class="muted">暂无任务</div>
      <div class="job-list" v-else>
        <article v-for="job in pipelineJobs" :key="job.id" class="job-card">
          <div class="job-head">
            <div>
              <strong>{{ job.id }}</strong>
              <span class="job-type">{{ job.type }}</span>
            </div>
            <button
              v-if="isRunningStatus(job.status)"
              @click="stopJob(job.id)"
            >
              停止
            </button>
          </div>
          <div class="job-meta">
            <span>状态: {{ job.status }}</span>
            <span>进程: {{ job.pid || '-' }}</span>
            <span>返回码: {{ job.return_code === null ? '-' : job.return_code }}</span>
            <span>开始: {{ job.started_at || '-' }}</span>
            <span>完成: {{ job.completed_at || '-' }}</span>
          </div>
          <pre v-if="job.log_tail" class="log-tail">{{ job.log_tail.join('\n') }}</pre>
          <div class="job-file" v-if="job.log_file">日志文件: {{ job.log_file }}</div>
        </article>
      </div>
    </section>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { RouterLink } from 'vue-router'
import {
  exportLabels,
  getDeployState,
  getPipelineJobs,
  setDeployState,
  startPipelineDownload,
  startPipelineGlobal,
  startPipelineTrain,
  stopPipelineJob
} from '../api/index.js'
import { getExportStats } from '../api/index.js'

const deployState = ref({ model_checkpoint: '', telegram_checkpoint: '' })
const deployInput = ref('')
const isUpdatingDeploy = ref(false)
const deployMessage = ref('')

const labelTotal = ref(0)
const isExporting = ref(false)
const exportError = ref('')
const jobsLoading = ref(false)
const pipelineJobs = ref([])
const jobsTimer = ref(null)

const isStartingTrain = ref(false)
const isStartingDownload = ref(false)
const isStartingGlobal = ref(false)
const trainError = ref('')
const downloadError = ref('')
const globalError = ref('')

const trainForm = reactive({
  labelsPath: 'labels.json',
  valRatio: 0.2,
  numFrames: 8,
  longVideoStrategy: 'expand',
  epochs: 10,
  batchSize: 16,
  learningRate: 0.0001,
  outputDir: 'checkpoints/frozen_clip',
  precision: 'fp32',
  scoreMin: '',
  scoreMax: '',
  clipBatchSize: 64,
  maxFrames: 32,
  maxLongFrames: 32,
  minLongFrames: '',
  numWorkers: 4,
  prefetchFactor: 4,
  persistentWorkers: true,
  saveEvery: 1,
})

const downloadForm = reactive({
  minScore: 7.0,
  chatBatchSize: 150,
  continuous: true,
  keepBelowThreshold: false,
  discoverChatTypes: '',
  breadthRounds: null,
  sessionName: '',
})

const globalForm = reactive({
  minScore: 7.0,
  chatBatchSize: 150,
  continuous: true,
  runBulkInfer: true,
  force: false,
  progressEvery: 100,
  pruneBelowThreshold: false,
  skipRebucket: false,
  dryRun: false,
  rebucketMode: 'symlink',
  skipDownload: false,
  breadthRounds: null,
})

function toNullableNumber(value) {
  if (value === '' || value === null || value === undefined) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function normalizePayload(payload) {
  const cleaned = {}
  Object.entries(payload).forEach(([key, value]) => {
    if (typeof value === 'string' && value.trim() === '') {
      cleaned[key] = null
      return
    }
    if (typeof value === 'number' && Number.isNaN(value)) {
      cleaned[key] = null
      return
    }
    if (value === null) {
      cleaned[key] = null
      return
    }
    cleaned[key] = value
  })
  return cleaned
}

function toTrainPayload() {
  const payload = {
    labels_path: trainForm.labelsPath,
    val_ratio: trainForm.valRatio,
    num_frames: trainForm.numFrames,
    long_video_strategy: trainForm.longVideoStrategy,
    epochs: trainForm.epochs,
    batch_size: trainForm.batchSize,
    learning_rate: trainForm.learningRate,
    output_dir: trainForm.outputDir,
    precision: trainForm.precision,
    num_workers: trainForm.numWorkers,
    prefetch_factor: trainForm.prefetchFactor,
    persistent_workers: trainForm.persistentWorkers,
    save_every: trainForm.saveEvery,
    clip_batch_size: trainForm.clipBatchSize,
    max_frames: trainForm.maxFrames,
    max_long_frames: toNullableNumber(trainForm.maxLongFrames),
    min_long_frames: toNullableNumber(trainForm.minLongFrames),
    score_min: toNullableNumber(trainForm.scoreMin),
    score_max: toNullableNumber(trainForm.scoreMax),
  }
  return normalizePayload(payload)
}

function toDownloadPayload() {
  return normalizePayload({
    min_score: downloadForm.minScore,
    chat_batch_size: downloadForm.chatBatchSize,
    continuous: downloadForm.continuous,
    keep_below_threshold: downloadForm.keepBelowThreshold,
    discover_chat_types: downloadForm.discoverChatTypes,
    breadth_rounds: downloadForm.breadthRounds,
    session_name: downloadForm.sessionName,
  })
}

function toGlobalPayload() {
  return normalizePayload({
    min_score: globalForm.minScore,
    chat_batch_size: globalForm.chatBatchSize,
    continuous: globalForm.continuous,
    run_bulk_infer: globalForm.runBulkInfer,
    force: globalForm.force,
    progress_every: globalForm.progressEvery,
    prune_below_threshold: globalForm.pruneBelowThreshold,
    skip_rebucket: globalForm.skipRebucket,
    dry_run: globalForm.dryRun,
    rebucket_mode: globalForm.rebucketMode,
    skip_download: globalForm.skipDownload,
    breadth_rounds: globalForm.breadthRounds,
  })
}

async function refreshDeployState() {
  const state = await getDeployState()
  deployState.value = state
  if (!deployInput.value) {
    deployInput.value = state.model_checkpoint
  }
}

async function updateDeployState() {
  if (!deployInput.value) {
    deployMessage.value = '请填写 checkpoint 路径'
    return
  }

  isUpdatingDeploy.value = true
  deployMessage.value = ''
  try {
    const response = await setDeployState(deployInput.value)
    deployState.value = {
      model_checkpoint: response.model_checkpoint,
      telegram_checkpoint: response.telegram_checkpoint,
    }
    deployMessage.value = '更新成功'
  } catch (error) {
    deployMessage.value = error?.response?.data?.detail || '更新失败'
  } finally {
    isUpdatingDeploy.value = false
  }
}

async function refreshExportStats() {
  try {
    const stats = await getExportStats()
    labelTotal.value = stats.total || 0
  } catch {
    labelTotal.value = 0
  }
}

function triggerBlobDownload(blob, filename) {
  const url = window.URL.createObjectURL(new Blob([blob]))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.URL.revokeObjectURL(url)
}

async function exportData() {
  exportError.value = ''
  isExporting.value = true
  try {
    const blob = await exportLabels('json')
    triggerBlobDownload(blob, `labels_${Date.now()}.json`)
    await refreshExportStats()
  } catch (error) {
    exportError.value = error?.response?.status ? `${error.response.status}: ${error.message}` : '导出失败'
  } finally {
    isExporting.value = false
  }
}

async function refreshJobs() {
  jobsLoading.value = true
  try {
    const data = await getPipelineJobs(true)
    pipelineJobs.value = data.jobs || []
  } catch (error) {
    pipelineJobs.value = []
    console.error('load jobs failed', error)
  } finally {
    jobsLoading.value = false
  }
}

function isRunningStatus(status) {
  return ['queued', 'running', 'stopping'].includes(status)
}

async function startTrain() {
  isStartingTrain.value = true
  trainError.value = ''
  try {
    await startPipelineTrain(toTrainPayload())
    await refreshJobs()
  } catch (error) {
    trainError.value = error?.response?.data?.detail || '启动训练失败'
  } finally {
    isStartingTrain.value = false
  }
}

async function startDownload() {
  isStartingDownload.value = true
  downloadError.value = ''
  try {
    await startPipelineDownload(toDownloadPayload())
    await refreshJobs()
  } catch (error) {
    downloadError.value = error?.response?.data?.detail || '启动下载失败'
  } finally {
    isStartingDownload.value = false
  }
}

async function startGlobal() {
  isStartingGlobal.value = true
  globalError.value = ''
  try {
    await startPipelineGlobal(toGlobalPayload())
    await refreshJobs()
  } catch (error) {
    globalError.value = error?.response?.data?.detail || '启动全链路失败'
  } finally {
    isStartingGlobal.value = false
  }
}

async function stopJob(jobId) {
  try {
    await stopPipelineJob(jobId)
    await refreshJobs()
  } catch (error) {
    console.error('stop job failed', error)
  }
}

onMounted(async () => {
  await Promise.all([
    refreshDeployState(),
    refreshExportStats(),
    refreshJobs(),
  ])
  jobsTimer.value = setInterval(refreshJobs, 4000)
})

onBeforeUnmount(() => {
  if (jobsTimer.value) {
    clearInterval(jobsTimer.value)
  }
})
</script>

<style scoped>
.pipeline-view {
  max-width: 1200px;
  margin: 1rem auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.pipeline-header {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.pipeline-section {
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 1rem;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.03);
}

.pipeline-section h2 {
  margin: 0 0 0.7rem 0;
}

.header-link {
  margin-top: 0.25rem;
}

.inline-form {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 0.5rem;
  align-items: center;
}

.inline-form input {
  min-width: 420px;
  padding: 0.45rem 0.6rem;
  border-radius: 8px;
}

.inline-form button,
.form-actions button,
.job-head button,
.jobs-header button,
.pipeline-section .form-grid ~ button {
  padding: 0.5rem 0.9rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 0.75rem;
}

.form-grid label {
  display: flex;
  flex-direction: column;
  font-size: 0.9rem;
  gap: 0.35rem;
}

.form-grid input,
.form-grid select {
  padding: 0.45rem 0.6rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
}

.form-actions {
  grid-column: 1 / -1;
  display: flex;
  gap: 0.5rem;
}

.jobs-header {
  margin-bottom: 0.6rem;
}

.job-list {
  display: grid;
  gap: 0.6rem;
}

.job-card {
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 10px;
  padding: 0.75rem;
}

.job-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.job-type {
  margin-left: 0.75rem;
  background: rgba(74, 144, 226, 0.2);
  padding: 0.15rem 0.45rem;
  border-radius: 999px;
  font-size: 0.85rem;
}

.job-meta {
  margin-top: 0.4rem;
  display: flex;
  gap: 0.8rem;
  flex-wrap: wrap;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.78);
}

.job-file {
  margin-top: 0.35rem;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.65);
}

.log-tail {
  margin-top: 0.65rem;
  max-height: 200px;
  overflow: auto;
  background: #0b0f1e;
  color: #d8dee9;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 8px;
  padding: 0.5rem;
  white-space: pre-wrap;
  font-size: 0.77rem;
}

.muted {
  color: rgba(255, 255, 255, 0.75);
}

.message {
  margin-top: 0.5rem;
  color: #79f5cb;
}

.error {
  margin-top: 0.5rem;
  color: #ff8888;
}
</style>
