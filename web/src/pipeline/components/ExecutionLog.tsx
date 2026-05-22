/**
 * Execution Log — bottom bar showing run messages and validation errors
 */

import { usePipeline } from '../PipelineStore';
import Icon from '../../components/common/Icon';

export default function ExecutionLog() {
  const { executionLog, validation, isRunning } = usePipeline();

  const hasErrors = validation && !validation.valid;

  return (
    <div className="pipeline-log">
      <div className="pipeline-log__header">
        <Icon name="terminal" size={12} />
        <span>Execution Log</span>
        {isRunning && <span className="pipeline-log__spinner" />}
        {hasErrors && (
          <span className="pipeline-log__error-badge">
            {validation.errors.filter(e => e.type === 'error').length} errors
          </span>
        )}
      </div>
      <div className="pipeline-log__body">
        {validation && validation.errors.length > 0 && (
          <>
            {validation.errors.map((err, i) => (
              <div key={i} className={`pipeline-log__line pipeline-log__line--${err.type}`}>
                {err.type === 'error' ? '✗' : '⚠'} {err.message}
                {err.nodeId && <span className="pipeline-log__node-ref"> ({err.nodeId})</span>}
              </div>
            ))}
          </>
        )}
        {executionLog.map((msg, i) => (
          <div
            key={i}
            className={`pipeline-log__line ${msg.startsWith('✓') ? 'pipeline-log__line--success' : msg.startsWith('✗') ? 'pipeline-log__line--error' : msg.startsWith('⚠') ? 'pipeline-log__line--warning' : ''}`}
          >
            {msg}
          </div>
        ))}
        {executionLog.length === 0 && !validation && (
          <div className="pipeline-log__placeholder">Ready. Click Run to execute the pipeline.</div>
        )}
      </div>
    </div>
  );
}
