# evalsift
 
**Which of your agent runs actually measured the model and what do the valid ones say?**
 
Agent evaluations fail in two different ways, and most pipelines blur them. Some runs are
*invalid*: a tool crashed, the API dropped, no score was written. They tell you nothing about
the model and should be excluded. Others are *valid but rough*: a malformed tool call, a
loop, a truncated answer. Those are real model behaviour and worth reading. Counting an
infrastructure crash as a model failure (or silently dropping it) quietly changes your
headline number.
 
`evalsift` reads [Inspect](https://inspect.aisi.org.uk/) logs, separates the two, and reports
rates on valid runs only, with Wilson confidence intervals and an explicit count of what was
excluded.

