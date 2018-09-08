import './index.pug'

import 'semantic-ui-offline/semantic.css'
import './app.sass'

import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'

window.$ = window.jQuery = require('jquery')
require('semantic-ui-offline/semantic.js')

const MODEL_URL = './model/tensorflowjs_model.pb'
const WEIGHTS_URL = './model/weights_manifest.json'

const video = document.getElementsByTagName('video')[0]
const canvas = document.getElementsByTagName('canvas')[0]

const width = 320
const height = 240

let faceCollection = []
let faceId = null

let isCapture = false
let isValidate = false
let captureProcess = null

video.setAttribute('width', width)
video.setAttribute('height', height)
canvas.setAttribute('width', width)
canvas.setAttribute('height', height)

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia

loadFrozenModel(MODEL_URL, WEIGHTS_URL).then(main)


function shapshot() {
  const context = canvas.getContext('2d')
  context.drawImage(video, 0, 0, width, height)
  return tf.fromPixels(context.getImageData(220, 140, 200, 200))
}

function main(model) {
  const successCallback = (stream) => {
    video.srcObject = stream
  }

  const errorCallback = (error) => {
    console.log("navigator.getUserMedia error: ", error)
  }

  navigator.getUserMedia({audio: false, video: true}, successCallback, errorCallback)

  $('#capture').click(e => {

    $(e.currentTarget).toggleClass('teal').toggleClass('red')
    isCapture = !isCapture

    if (isCapture) {
      $(e.currentTarget).text('Stop')
      captureProcess = setInterval(() => faceCollection.push(shapshot()), 500)
    } else {
      clearInterval(captureProcess)


      $(e.currentTarget).text('Build Your Face ID ... ').ready(() => {

        const res = model.execute({
          'squeeze_net/face_input': tf.cast(tf.transpose(tf.stack(faceCollection), [0, 2, 1, 3]), 'float32')
        })

        faceId = res.dataSync()

        // $(e.currentTarget).text('Capture Your Face ID')
        $(e.currentTarget).text(faceId.length / 128)
        console.log(faceId.length / 128)

        faceCollection.length = 0
      })

    }
  })

  $('#validate').click(e => {
    if (!faceId || faceId.length === 0) {
      alert('You don\'t have face id.')
    }

    if (isCapture) {
      return
    }

    $(e.currentTarget).toggleClass('blue').toggleClass('red')
    isValidate = !isValidate

    if (isValidate) {
      $(e.currentTarget).text('Stop validate').ready(() => {

        captureProcess = setInterval(() => {
          const res = model.execute({
            'squeeze_net/face_input': tf.transpose(shapshot().expandDims(), [0, 2, 1, 3]).asType('float32')
          })
          const target = res.dataSync()

          let dist = 0
          let vote = 0

          for (let i = 0; i < faceId.length; i++) {
            if (i % 128 === 0 && i != 0) {
              console.log(dist)
              vote += Math.sqrt(dist) < 0.8 ? 1 : 0
              dist = 0
            }

            dist += Math.pow(faceId[i] - target[i % 128], 2)
          }
          console.log(vote)

          if (vote >= faceId.length / 128 * 0.8) {
            $('#validate').click()
            $('#lock').transition('scale')
            $('#unlock').transition('scale')
          }
        }, 500)
      })

    } else {
      clearInterval(captureProcess)
      $(e.currentTarget).text('Validate')
    }
  })

  $('#relock').click(e => {
    $('#lock').transition('scale')
    $('#unlock').transition('scale')
  })
}
