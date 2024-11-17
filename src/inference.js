const tfjs = require('@tensorflow/tfjs-node')
function loadModel() {
  // const modelUrl = "\\wsl.localhost\Ubuntu-22.04\home\rifkifiransah\bangkit\ml-web-server\models\model.json"
  const modelUrl = "file://models/model.json"
  return tfjs.loadLayersModel(modelUrl)
}

function predict(model, imageBuffer) {
  const tensor = tfjs.node
    .decodeJpeg(imageBuffer)
    .resizeNearestNeighbor([150, 150])
    .expandDims()
    .toFloat()
  
  return model.predict(tensor).data()
}

module.exports = { loadModel, predict };