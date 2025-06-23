import traceback
import numpy as np
import os
import json
from flask import Flask, request, jsonify
from flask_mysqldb import MySQL
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import db_config
from utils.face_utils import obtener_embeddings_lbp as obtener_embeddings

# Función de similitud de coseno
def similitud_coseno(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Evitar división por cero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))

# Crear la app Flask
app = Flask(__name__)
CORS(app)

# Configuración de base de datos
app.config['MYSQL_HOST'] = db_config['host']
app.config['MYSQL_USER'] = db_config['user']
app.config['MYSQL_PASSWORD'] = db_config['password']
app.config['MYSQL_DB'] = db_config['database']

# Inicializar conexión
mysql = MySQL(app)

# Configurar la ruta de uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta raíz
@app.route("/")
def index():
    return "Backend funcionando correctamente."


# Registrar usuario con imagen
@app.route("/registrar_usuario", methods=["POST"])
def registrar_usuario():
    try:
        # Verificar que se recibieron datos de formulario
        if not request.form:
            return jsonify({"mensaje": "No se recibieron datos del formulario"}), 400
            
        # Obtener datos del formulario con get() para evitar KeyError
        nombre = request.form.get('nombre')
        apellido = request.form.get('apellido')
        codigo_unico = request.form.get('codigo_unico')
        email = request.form.get('email')
        requisitoriado = request.form.get('requisitoriado', 'false').lower() == 'true'
        
        # Validar campos obligatorios
        if not all([nombre, apellido, codigo_unico, email]):
            return jsonify({"mensaje": "Faltan campos requeridos", 
                          "detalles": {
                              "nombre": bool(nombre),
                              "apellido": bool(apellido),
                              "codigo_unico": bool(codigo_unico),
                              "email": bool(email)
                          }}), 400

        # Verificar si se envió una imagen
        if 'imagen' not in request.files:
            return jsonify({"mensaje": "No se proporcionó imagen"}), 400
            
        imagen = request.files['imagen']
        if imagen.filename == '':
            return jsonify({"mensaje": "Nombre de imagen no válido"}), 400

        # Registrar usuario en la base de datos
        cursor = mysql.connection.cursor()
        try:
            # Verificar si el código único ya existe
            cursor.execute("SELECT id FROM usuarios WHERE codigo_unico = %s", (codigo_unico,))
            if cursor.fetchone():
                return jsonify({"mensaje": "El código único ya está registrado"}), 400

            sql = """INSERT INTO usuarios (nombre, apellido, codigo_unico, email, requisitoriado)
                     VALUES (%s, %s, %s, %s, %s)"""
            cursor.execute(sql, (nombre, apellido, codigo_unico, email, requisitoriado))
            usuario_id = cursor.lastrowid
            
            # Crear carpeta para el usuario
            carpeta_usuario = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{usuario_id}")
            os.makedirs(carpeta_usuario, exist_ok=True)
            
            # Guardar imagen
            filename = secure_filename(f"user_{usuario_id}_{imagen.filename}")
            ruta_guardado = os.path.join(carpeta_usuario, filename)
            imagen.save(ruta_guardado)
            
            # Leer la imagen guardada para extraer embeddings
            with open(ruta_guardado, 'rb') as f:
                imagen_bytes = f.read()
            
            embeddings = obtener_embeddings(imagen_bytes)
            if embeddings is None:
                raise Exception("No se detectaron características faciales en la imagen")
            
            # Guardar información de la imagen en la base de datos
            ruta_relativa = os.path.join(f"user_{usuario_id}", filename)
            sql = """INSERT INTO imagenes (usuario_id, imagen_path, embeddings)
                     VALUES (%s, %s, %s)"""
            cursor.execute(sql, (usuario_id, ruta_relativa, json.dumps(embeddings)))
            mysql.connection.commit()
            
            return jsonify({
                "mensaje": "Usuario registrado exitosamente",
                "id_usuario": usuario_id,
                "nombre": nombre,
                "apellido": apellido
            }), 200
            
        except Exception as e:
            mysql.connection.rollback()
            print("Error en transacción de registro:", str(e))
            traceback.print_exc()
            return jsonify({"mensaje": f"Error en el registro: {str(e)}"}), 500
            
        finally:
            cursor.close()

    except Exception as e:
        print("Error general en registrar_usuario:", str(e))
        traceback.print_exc()
        return jsonify({"mensaje": "Error interno al procesar el registro"}), 500

# Agregar imagen a usuario existente
@app.route("/agregar_imagen/<int:usuario_id>", methods=["POST"])
def agregar_imagen(usuario_id):
    try:
        # Verificar que el usuario existe
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT id FROM usuarios WHERE id = %s", (usuario_id,))
        if not cursor.fetchone():
            return jsonify({"mensaje": "Usuario no encontrado"}), 404

        # Verificar imagen
        if 'imagen' not in request.files:
            return jsonify({"mensaje": "No se proporcionó imagen"}), 400
            
        imagen = request.files['imagen']
        if imagen.filename == '':
            return jsonify({"mensaje": "Nombre de imagen no válido"}), 400

        # Crear carpeta si no existe
        carpeta_usuario = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{usuario_id}")
        os.makedirs(carpeta_usuario, exist_ok=True)

        # Guardar imagen
        filename = secure_filename(f"extra_{usuario_id}_{imagen.filename}")
        ruta_guardado = os.path.join(carpeta_usuario, filename)
        imagen.save(ruta_guardado)

        # Procesar imagen
        with open(ruta_guardado, 'rb') as f:
            imagen_bytes = f.read()

        embeddings = obtener_embeddings(imagen_bytes)
        if embeddings is None:
            return jsonify({"mensaje": "No se detectaron características faciales"}), 400

        # Guardar en base de datos
        ruta_relativa = os.path.join(f"user_{usuario_id}", filename)
        sql = """INSERT INTO imagenes (usuario_id, imagen_path, embeddings)
                 VALUES (%s, %s, %s)"""
        cursor.execute(sql, (usuario_id, ruta_relativa, json.dumps(embeddings)))
        mysql.connection.commit()

        return jsonify({
            "mensaje": "Imagen agregada exitosamente",
            "imagen_path": ruta_relativa
        }), 200

    except Exception as e:
        print("Error en agregar_imagen:", str(e))
        traceback.print_exc()
        return jsonify({"mensaje": f"Error al agregar imagen: {str(e)}"}), 500
        
    finally:
        cursor.close()

# Listar usuarios
@app.route("/listar_usuarios", methods=["GET"])
def listar_usuarios():
    try:
        cursor = mysql.connection.cursor()
        cursor.execute("""
            SELECT id, nombre, apellido, codigo_unico, email, 
                   requisitoriado, fecha_registro 
            FROM usuarios
            ORDER BY fecha_registro DESC
        """)
        
        # Convertir resultados a lista de diccionarios
        columnas = [col[0] for col in cursor.description]
        usuarios = [dict(zip(columnas, fila)) for fila in cursor.fetchall()]
        
        # Convertir booleanos
        for usuario in usuarios:
            usuario['requisitoriado'] = bool(usuario['requisitoriado'])
            
        return jsonify(usuarios), 200

    except Exception as e:
        print("Error en listar_usuarios:", str(e))
        return jsonify({"mensaje": "Error al obtener lista de usuarios"}), 500
        
    finally:
        cursor.close()

# Reconocer usuario
@app.route("/reconocer_usuario", methods=["POST"])
def reconocer_usuario():
    try:
        # Verificar imagen
        if 'imagen' not in request.files:
            return jsonify({"mensaje": "No se proporcionó imagen"}), 400
            
        imagen = request.files['imagen']
        if imagen.filename == '':
            return jsonify({"mensaje": "Nombre de imagen no válido"}), 400

        # Guardar temporalmente y procesar imagen
        filename = secure_filename(f"temp_{imagen.filename}")
        ruta_temporal = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagen.save(ruta_temporal)

        with open(ruta_temporal, 'rb') as f:
            imagen_bytes = f.read()
            
        emb_ext = obtener_embeddings(imagen_bytes)
        os.remove(ruta_temporal)  # Eliminar temporal

        if emb_ext is None:
            return jsonify({"mensaje": "No se detectó un rostro en la imagen"}), 400

        # Buscar coincidencias en la base de datos
        cursor = mysql.connection.cursor()
        cursor.execute("""
            SELECT i.embeddings, i.imagen_path, u.id, u.nombre, 
                   u.apellido, u.codigo_unico, u.requisitoriado
            FROM imagenes i
            JOIN usuarios u ON i.usuario_id = u.id
        """)
        
        umbral_similitud = 0.985
        cantidad_minima = 2
        candidatos = {}

        for fila in cursor.fetchall():
            try:
                emb_guardado = json.loads(fila[0])
                if len(emb_guardado) != len(emb_ext):
                    continue

                similitud = similitud_coseno(emb_ext, emb_guardado)
                if similitud >= umbral_similitud:
                    usuario_id = fila[2]
                    if usuario_id not in candidatos:
                        candidatos[usuario_id] = {
                            "nombre": fila[3],
                            "apellido": fila[4],
                            "codigo_unico": fila[5],
                            "requisitoriado": bool(fila[6]),
                            "similitudes": [],
                            "imagenes": []
                        }
                    candidatos[usuario_id]["similitudes"].append(similitud)
                    candidatos[usuario_id]["imagenes"].append(fila[1])
            except Exception as e:
                print(f"Error procesando embeddings para imagen {fila[1]}: {str(e)}")
                continue

        # Encontrar mejor coincidencia
        mejor_usuario = None
        max_coincidencias = 0

        for uid, data in candidatos.items():
            if len(data["similitudes"]) >= cantidad_minima:
                promedio = sum(data["similitudes"]) / len(data["similitudes"])
                if len(data["similitudes"]) > max_coincidencias:
                    max_coincidencias = len(data["similitudes"])
                    mejor_usuario = {
                        "usuario_id": uid,
                        "nombre": data["nombre"],
                        "apellido": data["apellido"],
                        "codigo_unico": data["codigo_unico"],
                        "similitud_promedio": round(promedio, 4),
                        "requisitoriado": data["requisitoriado"],
                        "imagen_referencia": data["imagenes"][0],
                        "coincidencias": len(data["similitudes"])
                    }

        if mejor_usuario:
            if mejor_usuario["requisitoriado"]:
                mejor_usuario["alerta"] = True
                mejor_usuario["mensaje_alerta"] = "¡ALERTA! Usuario requisitoriado detectado."
            return jsonify(mejor_usuario), 200

        # No se encontró coincidencia
        carpeta_nuevo = os.path.join(app.config['UPLOAD_FOLDER'], "nuevo_usuario")
        os.makedirs(carpeta_nuevo, exist_ok=True)
        nueva_ruta = os.path.join(carpeta_nuevo, secure_filename(imagen.filename))
        imagen.save(nueva_ruta)

        return jsonify({
            "mensaje": "No se reconoció al usuario. Captura más imágenes para registro.",
            "capturas_pendientes": True,
            "ruta_imagen": os.path.join("nuevo_usuario", secure_filename(imagen.filename))
        }), 200

    except Exception as e:
        print("Error en reconocer_usuario:", str(e))
        traceback.print_exc()
        return jsonify({"mensaje": "Error en el reconocimiento facial"}), 500
        
    finally:
        cursor.close()

# Ejecutar la app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)