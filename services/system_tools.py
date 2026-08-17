import psutil


def get_cpu_usage():
    return psutil.cpu_percent(interval=1)


def get_memory_usage():
    memory = psutil.virtual_memory()

    return {
        "total_gb": round(memory.total / (1024 ** 3), 2),
        "used_gb": round(memory.used / (1024 ** 3), 2),
        "available_gb": round(memory.available / (1024 ** 3), 2),
        "percent": memory.percent,
    }


def get_disk_usage():
    disk = psutil.disk_usage("/")

    return {
        "total_gb": round(disk.total / (1024 ** 3), 2),
        "used_gb": round(disk.used / (1024 ** 3), 2),
        "free_gb": round(disk.free / (1024 ** 3), 2),
        "percent": disk.percent,
    }


def get_system_status():
    return {
        "cpu_percent": get_cpu_usage(),
        "memory": get_memory_usage(),
        "disk": get_disk_usage(),
    }


if __name__ == "__main__":
    print("CPU:", get_cpu_usage(), "%")
    print("Memory:", get_memory_usage())
    print("Disk:", get_disk_usage())
