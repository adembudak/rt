#include <algorithm> // for std::clamp
#include <array>
#include <cmath> // for std::pow, std::sqrt
#include <cstdint>
#include <functional> // for std::function
#include <limits>
#include <tuple>
#include <vector>

#define STBI_ONLY_PNG // decode only for PNG format
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace {

struct point {
  double x{}, y{}, z{};
};

struct vec3d {
  double x{}, y{}, z{};

  double &operator[](const std::size_t i) { return i == 0 ? x : i == 1 ? y : z; }
  const double &operator[](const std::size_t i) const { return i == 0 ? x : i == 1 ? y : z; }
};

template <typename T, std::size_t N> using matrixNxN = std::array<std::array<T, N>, N>;

using matrix3x3d = matrixNxN<double, 3>;

inline vec3d operator*(const matrix3x3d m, const vec3d v) {
  vec3d result;

  for (std::size_t i = 0; i < 3; i++) {
    for (std::size_t j = 0; j < 3; j++) {
      result[i] += v[j] * m[i][j];
    }
  }

  return result;
}

inline vec3d operator-(const point p, const point q) {
  return vec3d{p.x - q.x, p.y - q.y, p.z - q.z};
}

inline point operator+(const vec3d v, const point p) {
  return point{v.x + p.x, v.y + p.y, v.z + p.z};
}

inline vec3d operator+(const vec3d v, const vec3d w) {
  return vec3d{v.x + w.x, v.y + w.y, v.z + w.z};
}

inline vec3d operator*(const double k, const vec3d v) { return vec3d{k * v.x, k * v.y, k * v.z}; }

inline vec3d operator-(const vec3d v) { return -1.0 * v; }

inline vec3d operator-(const vec3d v, const vec3d w) { return v + (-1.0 * w); }

inline bool operator==(const vec3d v, const vec3d w) {
  return v.x == w.x && v.y == w.y && v.z == w.z;
}

inline bool operator==(const point p, const point q) {
  return p.x == q.x && p.y == q.y && p.z == q.z;
}

inline double length(const vec3d v) { return sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z)); }

inline vec3d normalize(const vec3d v) { return (1.0 / length(v)) * v; }

inline double dot(const vec3d v, const vec3d w) { return (v.x * w.x) + (v.y * w.y) + (v.z * w.z); }

inline vec3d cross(const vec3d v, const vec3d w) {
  vec3d r;
  r.x = v.y * w.z - v.z * w.y;
  r.y = v.z * w.x - v.x * w.z;
  r.z = v.x * w.y - v.y * w.x;

  return r;
}

constexpr const size_t canvas_width = 600;
constexpr const size_t canvas_height = 600;
constexpr const size_t channels = 4; // r,g,b,a

constexpr const double viewport_width = 1.0;
constexpr const double viewport_height = 1.0;
constexpr const double viewport_distance = 1.0;

constexpr const double inf = std::numeric_limits<double>::max();

using canvas_t = uint8_t[canvas_width][canvas_height][channels];
using point_t = point;

struct color_t {
  double r{}, g{}, b{}, a{};
};

static constexpr const color_t white{255, 255, 255, 255};
static constexpr const color_t red{255, 0, 0, 255};
static constexpr const color_t green{0, 255, 0, 255};
static constexpr const color_t blue{0, 0, 255, 255};
static constexpr const color_t yellow{255, 255, 0, 255};

color_t operator*(const color_t c, const double k) {
  return color_t{k * c.r, k * c.g, k * c.g, 255.0};
}

color_t operator+(const color_t c1, const color_t c2) {
  return color_t{c1.r + c2.r, c1.g + c2.g, c1.b + c2.b, 255.0};
}

enum class light_type { ambient, directional, point };

struct light {
  light_type type{};
  double intensity{};
  point_t position{};
};

struct sphere {
  double radius{};
  point_t center{};
  color_t color{};
  double specular{};
  double reflective{};
};

} // namespace

int main() {
  canvas_t canvas;

  const sphere sphere_1{/*radius*/ 1.0, /*center*/ point_t{0, -1, 3}, red, /*specular*/ 10,
                        /*reflective*/ 0.2};
  const sphere sphere_2{1.0, point_t{2, 0, 4}, blue, 100, 0.3};
  const sphere sphere_3{1.0, point_t{-2, 0, 4}, green, 10, 0.4};
  const sphere sphere_4{5000.0, point_t{0, -5001, 0}, yellow, 20, 0.5};
  const std::vector spheres{sphere_1, sphere_2, sphere_3, sphere_4};

  const light light_1{light_type::ambient, 0.2, {}};
  const light light_2{light_type::point, 0.6, point_t{2, 1, 0}};
  const light light_3{light_type::directional, 0.2, point_t{1, 4, 4}};
  const std::vector lights{light_1, light_2, light_3};

  auto intersectRaySphere = [=](const point_t O, const vec3d D, const sphere s) {
    const vec3d OC = O - s.center;

    const double a = dot(D, D);
    const double b = 2 * dot(OC, D);

    const double r = s.radius;
    const double c = dot(OC, OC) - r * r;

    const double discriminant = b * b - 4 * a * c;

    if (discriminant < 0) {
      return std::tuple(inf, inf);
    }

    const double t1 = (-b + sqrt(discriminant)) / (2 * a);
    const double t2 = (-b - sqrt(discriminant)) / (2 * a);

    return std::tuple{t1, t2};
  };

  auto closestIntersection = [&](const point_t O, const vec3d D, const double min,
                                 const double max) {
    double closest_t = inf;
    sphere closest_sphere;
    bool sphere_changed = false;

    for (const sphere &sp : spheres) {
      const auto [t1, t2] = intersectRaySphere(O, D, sp);
      if ((t1 > min && t1 < max) && t1 < closest_t) {
        closest_t = t1;
        closest_sphere = sp;
        sphere_changed = true;
      }

      if ((t2 > min && t2 < max) && t2 < closest_t) {
        closest_t = t2;
        closest_sphere = sp;
        sphere_changed = true;
      }
    }
    return std::tuple{closest_sphere, closest_t, sphere_changed};
  };

  auto reflectRay = [](const vec3d R, const vec3d N) { return (2.0 * dot(N, R) * N) - R; };

  auto computeLighting = [&](const point_t P, const vec3d N, const vec3d V, const double specular) {
    double total_intensity = 0;

    for (const auto light : lights) {
      if (light.type == light_type::ambient) {
        total_intensity += light.intensity;
      } else {
        vec3d L;
        double t_max;
        const double epsilon = 0.001;

        if (light.type == light_type::point) {
          L = light.position - P;
          t_max = 1;
        } else {
          L = light.position - point_t{0, 0, 0};
          t_max = inf;
        }

        const auto [shadow_sphere, shadow_t, sphere_changed] =
            closestIntersection(P, L, epsilon, t_max);

        if (!sphere_changed) {
          continue;
        }

        if (const double n_dot_l = dot(N, L); n_dot_l > 0) { // diffuse reflection
          total_intensity += (light.intensity * n_dot_l) / (length(N) * length(L));
        }

        if (const vec3d R = reflectRay(L, N); specular != -1) { // specular reflection
          if (double r_dot_v = dot(R, V); r_dot_v > 0) {
            total_intensity += light.intensity + pow(r_dot_v / (length(R) * length(V)), specular);
          }
        }
      }
    }
    return total_intensity;
  };

  std::function<color_t(const point_t, const vec3d, const double, const double, int)> traceRay =
      [&](const point_t O, const vec3d D, const double min, const double max,
          int recursion_depth) -> color_t {
    const auto [closest_sphere, closest_t, sphere_changed] = closestIntersection(O, D, min, max);

    if (!sphere_changed) {
      return white;
    }

    const point_t P = (closest_t * D) + O;
    const vec3d N = normalize(P - closest_sphere.center);

    const vec3d V = -D;
    const color_t local_color =
        closest_sphere.color * computeLighting(P, N, V, closest_sphere.specular);

    if (const double r = closest_sphere.reflective; r <= 0 || recursion_depth <= 0) {
      return local_color;
    }

    const double epsilon = 0.001;
    const vec3d R = reflectRay(-D, N);
    color_t reflected_color = traceRay(P, R, epsilon, inf, recursion_depth - 1);

    return local_color * (1 - closest_sphere.reflective) +
           reflected_color * closest_sphere.reflective;
  };

  auto canvas2viewport = [=](const int x, const int y) {
    const double Vx = (x * viewport_width) / canvas_width;
    const double Vy = (y * viewport_height) / canvas_height;

    return vec3d{Vx, Vy, viewport_distance};
  };

  // x in [-300, 300)
  // y in [-300, 300)
  auto put_pixel = [&](const int x, const int y, const color_t p) {
    const size_t sx = (canvas_width / 2) + x;
    const size_t sy = ((canvas_height / 2) - y) - 1;

    canvas[sy][sx][0] = std::clamp(p.r, 0.0, 255.0);
    canvas[sy][sx][1] = std::clamp(p.g, 0.0, 255.0);
    canvas[sy][sx][2] = std::clamp(p.b, 0.0, 255.0);
    canvas[sy][sx][3] = 255;
  };

  matrix3x3d camera_rotation = {{{0.7071, 0, -0.7071}, {0.0, 1.0, 0.0}, {0.7071, 0, 0.7071}}};
  const point_t O{3.0, 0.0, 1.0}; // camera position

  int number_of_recursions = 6;
  for (int y = -300; y < 300; ++y) {
    for (int x = -300; x < 300; ++x) {
      const vec3d D = canvas2viewport(x, y);
      const vec3d direction = camera_rotation * D;
      const color_t c = traceRay(O, direction, 1.0, inf, number_of_recursions);
      put_pixel(x, y, c);
    }
  }

  int ret = stbi_write_png("output.png", canvas_width, canvas_height, channels, canvas,
                           canvas_width * channels);
  if (!ret) {
    return ret;
  }
}
