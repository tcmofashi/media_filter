const fs = require('fs');
let code = fs.readFileSync('src/components/LabelPanel.vue', 'utf8');

code = code.replace(/const existingLabel = computed\(\(\) => \{[\s\S]*?\}\)/, `const existingLabel = computed(() => {
  if (!props.selectedFile || !props.labels) return null
  const rawLabels = toRaw(props.labels)
  // iterate in reverse because newest labels are often at the end
  for (let i = rawLabels.length - 1; i >= 0; i--) {
    if (rawLabels[i].media_path === props.selectedFile.path) return rawLabels[i]
  }
  return null
})`);

fs.writeFileSync('src/components/LabelPanel.vue', code);
