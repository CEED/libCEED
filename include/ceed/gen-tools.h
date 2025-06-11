#include <ceed.h>
#include <sstream>

class Tab {
 private:
  CeedInt       _num_tabs{0};
  const CeedInt _width{2};

  template <class OStream>
  friend OStream &operator<<(OStream &os, const Tab &tab);

 public:
  Tab &push() {
    _num_tabs++;
    return *this;
  }
  Tab &pop() {
    if (_num_tabs > 0) _num_tabs--;
    return *this;
  }
};

template <class OStream>
OStream &operator<<(OStream &os, const Tab &tab) {
  os << std::string(tab._num_tabs * tab._width, ' ');
  return os;
}
