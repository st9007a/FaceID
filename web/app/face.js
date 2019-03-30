import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'

let model = null

export default {
  entry: 'mobile_net_v2/face_input',

  modelUrl: './model/tensorflowjs_model.pb',

  weightsUrl: './model/weights_manifest.json',

  init: function() {
    return loadFrozenModel(this.modelUrl, this.weightsUrl).then(res => model = res)
  },

  images2tensor: function(images) {
    return tf.tidy(() => tf.cast(tf.stack(images.map(el => tf.fromPixels(el))), 'float32'))
  },

  inference: function(images) {
    let obj = {}
    obj[this.entry] = this.images2tensor(images)

    const modelOutput = model.execute(obj)
    const result = modelOutput.dataSync()

    obj[this.entry].dispose()
    modelOutput.dispose()

    return result
  },

  transform: function(images) {
    const output = this.inference(images)
    const ids = []

    for (let i = 0; i < output.length; i += 128) {
      ids.push(output.slice(i, i + 128))
    }

    return ids
  },

  transformMore: function(images) {
    const batch = 8
    const ids = []

    for (let i = 0; i < images.length; i += batch) {
      const output = this.inference(images.slice(i, i + batch))

      for (let j = 0; j < output.length; j += 128) {
        ids.push(output.slice(j, j + 128))
      }
    }
    return ids
  }
}
