
def setup_logging(debug=False, disable=False):
    """Configure logging with timing, line numbers, and log rotation. Set disable=True to disable all logging."""
    import logging.handlers
    import os

    # Create logs directory
    logs_dir = "logs/"
    os.makedirs(logs_dir, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if disable:
        logger.setLevel(logging.CRITICAL + 1)
        return

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler for general logs
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, 'mimosa.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Separate handler for workflow execution logs
    workflow_handler = logging.handlers.RotatingFileHandler(
        os.path.join(logs_dir, 'workflows.log'),
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10
    )
    workflow_handler.setLevel(logging.INFO)
    workflow_handler.setFormatter(formatter)

    # Add workflow handler to specific loggers
    workflow_loggers = [
        'sources.core.evolution_engine',
        'sources.core.orchestrator',
        'sources.core.workflow_factory',
        'sources.core.workflow_runner',
        'sources.evaluation.evaluator'
    ]

    for logger_name in workflow_loggers:
        workflow_logger = logging.getLogger(logger_name)
        workflow_logger.addHandler(workflow_handler)