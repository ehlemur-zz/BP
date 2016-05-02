import time

class LemurTimer:
  def __init__(self, msg):
    self.msg = msg
    self.interrupted = False

  def __enter__(self):
    print "Started " + self.msg
    self.t = time.time()

  def __exit__(self, type_, value_, traceback_):
    if type_ is None:
      print "Finished " + self.msg + " " + self._format(time.time() - self.t)
    else:
      print "Interrupted " + self.msg + " after " + self._format(time.time() - self.t)

  def _format(self, t):
    if t >= 60:
      m = int(t / 60)
      s = int(t - 60 * m)
      return "%dm %ds" % (m, s)
    return "%.2fs" % t
