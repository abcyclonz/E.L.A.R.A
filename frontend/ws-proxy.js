const http = require('http')
const httpProxy = require('http-proxy')

const NEXT_ORIGIN = 'http://localhost:3002'
const ORCHESTRATOR_ORIGIN = 'http://localhost:8003'
const PORT = 3000

const proxy = httpProxy.createProxyServer({ ws: true })

proxy.on('error', (err, req, res) => {
  if (res && res.writeHead) {
    res.writeHead(502)
    res.end('Proxy error: ' + err.message)
  }
})

const server = http.createServer((req, res) => {
  proxy.web(req, res, { target: NEXT_ORIGIN })
})

server.on('upgrade', (req, socket, head) => {
  if (req.url === '/ws/audio') {
    proxy.ws(req, socket, head, { target: ORCHESTRATOR_ORIGIN })
  } else {
    socket.destroy()
  }
})

server.listen(PORT, () => {
  console.log(`WS proxy ready on :${PORT} (Next→3001, /ws/audio→8003)`)
})
