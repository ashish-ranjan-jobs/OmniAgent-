# OpenTelemetry Tracing

OpenTelemetry instruments services with traces composed of spans: each span names an operation, carries start/end timestamps, status, and attributes, and links to its parent for distributed causality. Context propagation (trace and span ids across HTTP or message headers) stitches cross-service calls into one trace. Semantic conventions standardize attribute names so backends can chart any instrumented component uniformly.
