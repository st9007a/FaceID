import 'libsvm-js/dist/browser/wasm/libsvm.wasm'

let SVM = null

function supportWASM() {
  try {
    if (typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function") {
      const module = new WebAssembly.Module(Uint8Array.of(0x0, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00))
      if (module instanceof WebAssembly.Module) {
        return new WebAssembly.Instance(module) instanceof WebAssembly.Instance
      }
    }
  } catch (e) {
    console.log(e)
  }
  return false
}

export default {

  model: null,

  init: function() {
    if (supportWASM() === true) {
      return require('libsvm-js/dist/browser/wasm/libsvm.js').then(lib => SVM = lib)
    } else {
      return require('libsvm-js/dist/browser/asm/libsvm.js').then(lib => SVM = lib)
    }
  },

  buildOCSVM: function(features) {
    this.model = new SVM({
      kernel: SVM.KERNEL_TYPES.RBF,
      type: SVM.SVM_TYPES.ONE_CLASS,
      gamma: 0.8,
      nu: 0.2,
    })

    const labels = []
    for (let i = 0; i < features.length; ++i) {
      labels.push(0)
    }

    this.model.train(features, labels)
  },

  predict: function(features) {
    return this.model.predictOne(features)
  },

}
