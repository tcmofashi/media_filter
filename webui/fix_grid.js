const fs = require('fs');
let code = fs.readFileSync('src/components/MediaGrid.vue', 'utf8');

// import toRaw
if (!code.includes('toRaw')) {
  code = code.replace(/import { ref, computed, watch, onUnmounted } from 'vue'/, "import { ref, computed, watch, onUnmounted, toRaw } from 'vue'");
}

code = code.replace(/const labelMap = computed\(\(\) => \{[\s\S]*?return map\n\}\)/, `const labelMap = computed(() => {
  const map = new Map()
  const rawLabels = toRaw(props.labels)
  for (let i = 0; i < rawLabels.length; i++) {
    map.set(rawLabels[i].media_path, rawLabels[i])
  }
  return map
})`);

fs.writeFileSync('src/components/MediaGrid.vue', code);
