const { spawn } = require('child_process');
const fs = require('fs');

const logFile = 'scratch_mcp_output.txt';
fs.writeFileSync(logFile, ''); // clear log

function log(msg) {
  fs.appendFileSync(logFile, msg + '\n');
  console.log(msg);
}

const proc = spawn('npx', [
  '-y',
  'mcp-remote',
  'https://stitch.googleapis.com/mcp',
  '--header',
  'X-Goog-Api-Key: AQ.Ab8RN6LBZjIZ9WQ4s8UxfVlVkECsu0ZntmQ7x9dpPJK4L1b-rQ'
]);

proc.stdout.on('data', (data) => {
  const text = data.toString();
  log('=== STDOUT ===\n' + text);
});

proc.stderr.on('data', (data) => {
  log('=== STDERR ===\n' + data.toString());
});

proc.on('close', (code) => {
  log('Child process exited with code ' + code);
});

// Step 1: Send the initialize request
const initReq = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2024-11-05",
    capabilities: {},
    clientInfo: {
      name: "test-client",
      version: "1.0.0"
    }
  }
};

log('Sending initialize request...');
proc.stdin.write(JSON.stringify(initReq) + '\n');

// Step 2: Send initialized notification
setTimeout(() => {
  const initializedNotification = {
    jsonrpc: "2.0",
    method: "notifications/initialized"
  };
  log('Sending initialized notification...');
  proc.stdin.write(JSON.stringify(initializedNotification) + '\n');
}, 1000);

// Step 3: Send tools/list request
setTimeout(() => {
  const toolsReq = {
    jsonrpc: "2.0",
    id: 2,
    method: "tools/list",
    params: {}
  };
  log('Sending tools/list request...');
  proc.stdin.write(JSON.stringify(toolsReq) + '\n');
}, 2000);

// Step 4: Send tools/call list_projects request
setTimeout(() => {
  const callReq = {
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: {
      name: "list_projects",
      arguments: {}
    }
  };
  log('Sending tools/call list_projects...');
  proc.stdin.write(JSON.stringify(callReq) + '\n');
}, 4500);

// Step 5: Exit after 12 seconds
setTimeout(() => {
  log('Terminating child process...');
  proc.kill();
  process.exit(0);
}, 12000);
